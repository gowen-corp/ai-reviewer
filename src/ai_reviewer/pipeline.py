"""Review pipeline for AI code reviewer.

This module implements the multi-stage review pipeline with support for
different review modes (light, standard, deep) and fail-fast logic.

Pipeline stages:
1. PRECHECK - Quick syntax and basic quality check
2. DIFF_ANALYSIS - Analyze changes in context  
3. CONTEXT_READ - Deep analysis with broader context
4. VERIFY - Final verification and aggregation

Modes:
- light: Only PRECHECK stage
- standard: PRECHECK + DIFF_ANALYSIS
- deep: All stages including aggregation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .llm_client import LLMClient
from .target_resolver import ReviewTarget
from .prompts import get_prompt_for_stage, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ReviewMode(Enum):
    """Review mode determining pipeline depth."""
    
    LIGHT = auto()      # 1 stage: PRECHECK
    STANDARD = auto()   # 2 stages: PRECHECK + DIFF_ANALYSIS
    DEEP = auto()       # 3+ stages: All stages with aggregation


class Verdict(Enum):
    """Review verdict."""
    
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class Finding:
    """Represents a single code review finding."""
    
    severity: str  # critical, major, minor
    category: str  # bug, security, performance, style, maintainability
    description: str
    line_number: int | None = None
    suggestion: str | None = None
    file_path: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line_number": self.line_number,
            "suggestion": self.suggestion,
            "file_path": self.file_path,
        }


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    
    stage_name: str
    verdict: Verdict
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    
    @property
    def has_critical_findings(self) -> bool:
        """Check if any findings are critical."""
        return any(f.severity == "critical" for f in self.findings)
    
    @property
    def critical_count(self) -> int:
        """Count of critical findings."""
        return sum(1 for f in self.findings if f.severity == "critical")
    
    @property
    def major_count(self) -> int:
        """Count of major findings."""
        return sum(1 for f in self.findings if f.severity == "major")


@dataclass
class ReviewResult:
    """Final result of the review pipeline."""
    
    target: ReviewTarget
    mode: ReviewMode
    overall_verdict: Verdict
    all_findings: list[Finding] = field(default_factory=list)
    stage_results: list[StageResult] = field(default_factory=list)
    blocked: bool = False
    blocked_at_stage: str | None = None
    summary: str = ""
    
    @property
    def critical_count(self) -> int:
        """Total count of critical findings."""
        return sum(1 for f in self.all_findings if f.severity == "critical")
    
    @property
    def major_count(self) -> int:
        """Total count of major findings."""
        return sum(1 for f in self.all_findings if f.severity == "major")
    
    @property
    def minor_count(self) -> int:
        """Total count of minor findings."""
        return sum(1 for f in self.all_findings if f.severity == "minor")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "verdict": self.overall_verdict.value,
            "blocked": self.blocked,
            "blocked_at_stage": self.blocked_at_stage,
            "summary": self.summary,
            "findings_count": len(self.all_findings),
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "findings": [f.to_dict() for f in self.all_findings],
            "stages": [
                {
                    "stage": sr.stage_name,
                    "verdict": sr.verdict.value,
                    "findings_count": len(sr.findings),
                    "summary": sr.summary,
                }
                for sr in self.stage_results
            ],
        }


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    pass


class FailFastError(PipelineError):
    """Raised when pipeline stops due to fail-fast condition."""
    
    def __init__(self, message: str, stage_result: StageResult) -> None:
        super().__init__(message)
        self.stage_result = stage_result


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse LLM JSON response.
    
    Args:
        response_text: Raw response text from LLM.
        
    Returns:
        Parsed JSON dictionary.
        
    Raises:
        ValueError: If response is not valid JSON.
    """
    # Try to extract JSON from response (handle markdown code blocks)
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM JSON response: %s", e)
        raise ValueError(f"Invalid JSON response: {e}") from e


def validate_response_structure(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize LLM response structure.
    
    Args:
        data: Parsed JSON response.
        
    Returns:
        Normalized response dictionary.
        
    Raises:
        ValueError: If required fields are missing.
    """
    # Check required fields
    if "verdict" not in data:
        raise ValueError("Missing required field: verdict")
    
    if "findings" not in data:
        data["findings"] = []
    
    # Validate verdict
    verdict_value = data["verdict"].upper()
    if verdict_value not in ("PASS", "WARN", "BLOCK"):
        logger.warning("Unknown verdict '%s', defaulting to WARN", data["verdict"])
        verdict_value = "WARN"
    
    data["verdict"] = verdict_value
    
    # Normalize findings
    normalized_findings = []
    for finding in data["findings"]:
        if not isinstance(finding, dict):
            continue
        
        normalized = {
            "severity": finding.get("severity", "minor"),
            "category": finding.get("category", "maintainability"),
            "description": finding.get("description", ""),
            "line_number": finding.get("line_number"),
            "suggestion": finding.get("suggestion", ""),
        }
        
        # Validate severity
        if normalized["severity"] not in ("critical", "major", "minor"):
            normalized["severity"] = "minor"
        
        # Validate category
        if normalized["category"] not in ("bug", "security", "performance", "style", "maintainability"):
            normalized["category"] = "maintainability"
        
        if normalized["description"]:
            normalized_findings.append(normalized)
    
    data["findings"] = normalized_findings
    
    return data


def findings_to_json(findings: list[Finding]) -> str:
    """Convert findings list to JSON string."""
    return json.dumps([f.to_dict() for f in findings], indent=2)


class ReviewPipeline:
    """Multi-stage code review pipeline.
    
    This class orchestrates the review process through multiple stages,
    with support for different modes and fail-fast behavior.
    
    Attributes:
        llm_client: LLM client for making API calls.
        mode: Review mode (light, standard, deep).
        fail_fast_threshold: Number of critical findings to trigger fail-fast.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        mode: ReviewMode = ReviewMode.STANDARD,
        fail_fast_threshold: int = 1,
    ) -> None:
        """Initialize the review pipeline.
        
        Args:
            llm_client: LLM client instance.
            mode: Review mode.
            fail_fast_threshold: Critical findings count to trigger fail-fast.
        """
        self.llm_client = llm_client
        self.mode = mode
        self.fail_fast_threshold = fail_fast_threshold
    
    def _get_stages_for_mode(self) -> list[str]:
        """Get list of stages to run based on mode."""
        if self.mode == ReviewMode.LIGHT:
            return ["precheck"]
        elif self.mode == ReviewMode.STANDARD:
            return ["precheck", "diff_analysis"]
        else:  # DEEP
            return ["precheck", "diff_analysis", "context_read", "verify"]
    
    async def _run_stage(
        self,
        stage_name: str,
        target: ReviewTarget,
        file_path: Path,
        file_content: str,
        previous_findings: list[Finding],
    ) -> StageResult:
        """Run a single review stage.
        
        Args:
            stage_name: Name of the stage.
            target: Review target.
            file_path: Path to the file being reviewed.
            file_content: Content of the file.
            previous_findings: Findings from previous stages.
            
        Returns:
            StageResult with findings and verdict.
        """
        # Get language from file extension
        language = file_path.suffix.lstrip(".") or "text"
        
        # Build prompt based on stage
        if stage_name == "precheck":
            prompt = get_prompt_for_stage(
                "precheck",
                language=language,
                code_content=file_content,
            )
        elif stage_name == "diff_analysis":
            # For now, use file content as both original and new
            # In future phases, this will use actual diff
            prompt = get_prompt_for_stage(
                "diff_analysis",
                file_path=str(file_path),
                language=language,
                original_code=file_content,
                new_code=file_content,
                diff_content="",
            )
        elif stage_name == "context_read":
            change_summary = "Code review in progress"
            prev_findings_str = findings_to_json(previous_findings) if previous_findings else "No previous findings"
            prompt = get_prompt_for_stage(
                "context_read",
                file_path=str(file_path),
                language=language,
                full_code=file_content,
                change_summary=change_summary,
                previous_findings=prev_findings_str,
            )
        elif stage_name == "verify":
            all_findings_str = findings_to_json(previous_findings) if previous_findings else "No findings"
            prompt = get_prompt_for_stage(
                "verify",
                file_path=str(file_path),
                all_findings_json=all_findings_str,
                review_mode=self.mode.name.lower(),
            )
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        # Call LLM
        response = await self.llm_client.chat_completion(
            messages=messages,
            model=self.llm_client.model,
            max_tokens=self.llm_client.timeout,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        
        # Parse and validate response
        try:
            parsed = parse_llm_response(response.content)
            validated = validate_response_structure(parsed)
        except ValueError as e:
            logger.error("Failed to parse/validate response: %s", e)
            # Return a safe default
            return StageResult(
                stage_name=stage_name,
                verdict=Verdict.WARN,
                findings=[],
                summary=f"Failed to parse LLM response: {e}",
                raw_response=response.content,
            )
        
        # Convert findings
        findings = []
        for f in validated["findings"]:
            findings.append(Finding(
                severity=f["severity"],
                category=f["category"],
                description=f["description"],
                line_number=f.get("line_number"),
                suggestion=f.get("suggestion"),
                file_path=str(file_path),
            ))
        
        verdict_str = validated.get("verdict", "WARN")
        verdict = Verdict[verdict_str]
        summary = validated.get("summary", "")
        
        return StageResult(
            stage_name=stage_name,
            verdict=verdict,
            findings=findings,
            summary=summary,
            raw_response=response.content,
        )
    
    async def review_file(
        self,
        target: ReviewTarget,
        file_path: Path,
    ) -> ReviewResult:
        """Run the review pipeline on a single file.
        
        Args:
            target: Review target containing metadata.
            file_path: Path to the file to review.
            
        Returns:
            ReviewResult with findings and verdict.
            
        Raises:
            FailFastError: If fail-fast condition is met.
        """
        # Read file content
        try:
            file_content = file_path.read_text(encoding="utf-8")
        except (IOError, OSError) as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return ReviewResult(
                target=target,
                mode=self.mode,
                overall_verdict=Verdict.WARN,
                blocked=False,
                summary=f"Failed to read file: {e}",
            )
        
        stages = self._get_stages_for_mode()
        stage_results: list[StageResult] = []
        all_findings: list[Finding] = []
        blocked = False
        blocked_at_stage: str | None = None
        
        for stage_name in stages:
            logger.info("Running stage: %s on %s", stage_name, file_path)
            
            result = await self._run_stage(
                stage_name=stage_name,
                target=target,
                file_path=file_path,
                file_content=file_content,
                previous_findings=all_findings,
            )
            
            stage_results.append(result)
            all_findings.extend(result.findings)
            
            # Check fail-fast condition
            if result.critical_count >= self.fail_fast_threshold:
                logger.warning(
                    "Fail-fast triggered: %d critical findings at stage %s",
                    result.critical_count,
                    stage_name,
                )
                blocked = True
                blocked_at_stage = stage_name
                break
            
            # Also check if verdict is BLOCK
            if result.verdict == Verdict.BLOCK:
                logger.warning("BLOCK verdict at stage %s", stage_name)
                blocked = True
                blocked_at_stage = stage_name
                break
        
        # Determine overall verdict
        if blocked:
            overall_verdict = Verdict.BLOCK
        elif any(f.severity == "critical" for f in all_findings):
            overall_verdict = Verdict.BLOCK
        elif any(f.severity == "major" for f in all_findings):
            overall_verdict = Verdict.WARN
        else:
            # Only minor findings or no findings = PASS
            overall_verdict = Verdict.PASS
        
        summary = f"Reviewed {file_path.name}: {len(all_findings)} findings"
        if stage_results:
            summary = stage_results[-1].summary or summary
        
        return ReviewResult(
            target=target,
            mode=self.mode,
            overall_verdict=overall_verdict,
            all_findings=all_findings,
            stage_results=stage_results,
            blocked=blocked,
            blocked_at_stage=blocked_at_stage,
            summary=summary,
        )
    
    async def review(self, target: ReviewTarget) -> list[ReviewResult]:
        """Run the review pipeline on all files in the target.
        
        Args:
            target: Review target with files to review.
            
        Returns:
            List of ReviewResult for each file.
        """
        results: list[ReviewResult] = []
        
        # Handle stdin content
        if target.source_type.name == "STDIN" and target.stdin_content:
            # For stdin, we create a synthetic result
            logger.info("Reviewing stdin content")
            
            # Create a temp file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(target.stdin_content)
                temp_path = Path(f.name)
            
            try:
                result = await self.review_file(target, temp_path)
                results.append(result)
            finally:
                temp_path.unlink()
            
            return results
        
        # Process each file
        for file_path in target.files:
            logger.info("Reviewing file: %s", file_path)
            result = await self.review_file(target, file_path)
            results.append(result)
            
            # Stop on first blocked result if fail-fast is aggressive
            if result.blocked and self.fail_fast_threshold == 1:
                logger.warning("Stopping after blocked file: %s", file_path)
                break
        
        return results


async def run_review(
    target: ReviewTarget,
    llm_client: LLMClient,
    mode: str = "standard",
    fail_fast_threshold: int = 1,
) -> list[ReviewResult]:
    """Convenience function to run a review.
    
    Args:
        target: Review target.
        llm_client: LLM client instance.
        mode: Review mode string (light, standard, deep).
        fail_fast_threshold: Critical findings to trigger fail-fast.
        
    Returns:
        List of ReviewResult objects.
    """
    mode_map = {
        "light": ReviewMode.LIGHT,
        "standard": ReviewMode.STANDARD,
        "deep": ReviewMode.DEEP,
    }
    
    review_mode = mode_map.get(mode.lower(), ReviewMode.STANDARD)
    
    pipeline = ReviewPipeline(
        llm_client=llm_client,
        mode=review_mode,
        fail_fast_threshold=fail_fast_threshold,
    )
    
    return await pipeline.review(target)
