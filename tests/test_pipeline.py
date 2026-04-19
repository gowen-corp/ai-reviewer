"""Unit tests for the review pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.pipeline import (
    ReviewMode,
    Verdict,
    Finding,
    StageResult,
    ReviewResult,
    ReviewPipeline,
    parse_llm_response,
    validate_response_structure,
    findings_to_json,
    run_review,
)
from ai_reviewer.llm_client import LLMClient, ChatCompletionResponse
from ai_reviewer.target_resolver import ReviewTarget, TargetSourceType
from ai_reviewer.token_tracker import TokenUsage


class TestFinding:
    """Tests for Finding dataclass."""
    
    def test_finding_creation(self):
        """Test creating a finding."""
        finding = Finding(
            severity="critical",
            category="bug",
            description="Null pointer exception",
            line_number=42,
            suggestion="Add null check",
            file_path="test.py",
        )
        
        assert finding.severity == "critical"
        assert finding.category == "bug"
        assert finding.description == "Null pointer exception"
        assert finding.line_number == 42
        assert finding.suggestion == "Add null check"
        assert finding.file_path == "test.py"
    
    def test_finding_to_dict(self):
        """Test converting finding to dictionary."""
        finding = Finding(
            severity="major",
            category="security",
            description="SQL injection risk",
            line_number=10,
            suggestion="Use parameterized queries",
            file_path="db.py",
        )
        
        result = finding.to_dict()
        
        assert result == {
            "severity": "major",
            "category": "security",
            "description": "SQL injection risk",
            "line_number": 10,
            "suggestion": "Use parameterized queries",
            "file_path": "db.py",
        }
    
    def test_finding_optional_fields(self):
        """Test finding with optional fields."""
        finding = Finding(
            severity="minor",
            category="style",
            description="Variable naming",
        )
        
        assert finding.line_number is None
        assert finding.suggestion is None
        assert finding.file_path is None


class TestStageResult:
    """Tests for StageResult dataclass."""
    
    def test_stage_result_creation(self):
        """Test creating a stage result."""
        findings = [
            Finding(severity="critical", category="bug", description="Bug 1"),
            Finding(severity="major", category="performance", description="Perf issue"),
        ]
        
        result = StageResult(
            stage_name="precheck",
            verdict=Verdict.BLOCK,
            findings=findings,
            summary="Found critical issues",
        )
        
        assert result.stage_name == "precheck"
        assert result.verdict == Verdict.BLOCK
        assert len(result.findings) == 2
    
    def test_has_critical_findings(self):
        """Test critical findings detection."""
        # With critical finding
        result = StageResult(
            stage_name="test",
            verdict=Verdict.BLOCK,
            findings=[Finding(severity="critical", category="bug", description="x")],
        )
        assert result.has_critical_findings is True
        
        # Without critical finding
        result = StageResult(
            stage_name="test",
            verdict=Verdict.PASS,
            findings=[Finding(severity="minor", category="style", description="x")],
        )
        assert result.has_critical_findings is False
    
    def test_critical_count(self):
        """Test counting critical findings."""
        result = StageResult(
            stage_name="test",
            verdict=Verdict.BLOCK,
            findings=[
                Finding(severity="critical", category="bug", description="1"),
                Finding(severity="critical", category="security", description="2"),
                Finding(severity="major", category="perf", description="3"),
            ],
        )
        assert result.critical_count == 2
    
    def test_major_count(self):
        """Test counting major findings."""
        result = StageResult(
            stage_name="test",
            verdict=Verdict.WARN,
            findings=[
                Finding(severity="major", category="bug", description="1"),
                Finding(severity="major", category="perf", description="2"),
                Finding(severity="minor", category="style", description="3"),
            ],
        )
        assert result.major_count == 2


class TestReviewResult:
    """Tests for ReviewResult dataclass."""
    
    def test_review_result_creation(self):
        """Test creating a review result."""
        target = ReviewTarget(
            files=[Path("test.py")],
            source_type=TargetSourceType.FILES,
        )
        
        result = ReviewResult(
            target=target,
            mode=ReviewMode.STANDARD,
            overall_verdict=Verdict.WARN,
            all_findings=[
                Finding(severity="major", category="bug", description="Issue"),
            ],
            blocked=False,
        )
        
        assert result.mode == ReviewMode.STANDARD
        assert result.overall_verdict == Verdict.WARN
        assert result.blocked is False
        assert len(result.all_findings) == 1
    
    def test_review_result_counts(self):
        """Test finding count properties."""
        target = ReviewTarget(files=[], source_type=TargetSourceType.FILES)
        
        result = ReviewResult(
            target=target,
            mode=ReviewMode.DEEP,
            overall_verdict=Verdict.BLOCK,
            all_findings=[
                Finding(severity="critical", category="bug", description="1"),
                Finding(severity="major", category="perf", description="2"),
                Finding(severity="major", category="security", description="3"),
                Finding(severity="minor", category="style", description="4"),
            ],
        )
        
        assert result.critical_count == 1
        assert result.major_count == 2
        assert result.minor_count == 1
    
    def test_review_result_to_dict(self):
        """Test converting review result to dictionary."""
        target = ReviewTarget(files=[Path("test.py")], source_type=TargetSourceType.FILES)
        
        finding = Finding(
            severity="critical",
            category="security",
            description="Vulnerability",
            line_number=5,
            suggestion="Fix it",
            file_path="test.py",
        )
        
        stage_result = StageResult(
            stage_name="precheck",
            verdict=Verdict.BLOCK,
            findings=[finding],
            summary="Blocked",
        )
        
        result = ReviewResult(
            target=target,
            mode=ReviewMode.LIGHT,
            overall_verdict=Verdict.BLOCK,
            all_findings=[finding],
            stage_results=[stage_result],
            blocked=True,
            blocked_at_stage="precheck",
            summary="Security issue found",
        )
        
        d = result.to_dict()
        
        assert d["verdict"] == "BLOCK"
        assert d["blocked"] is True
        assert d["blocked_at_stage"] == "precheck"
        assert d["summary"] == "Security issue found"
        assert d["findings_count"] == 1
        assert d["critical_count"] == 1
        assert len(d["findings"]) == 1
        assert len(d["stages"]) == 1


class TestParseLLMResponse:
    """Tests for parse_llm_response function."""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        response = json.dumps({
            "verdict": "PASS",
            "findings": [],
            "summary": "All good",
        })
        
        result = parse_llm_response(response)
        
        assert result["verdict"] == "PASS"
        assert result["findings"] == []
        assert result["summary"] == "All good"
    
    def test_parse_json_with_markdown_block(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = '''```json
{
    "verdict": "WARN",
    "findings": [{"severity": "minor", "category": "style", "description": "x"}]
}
```'''
        
        result = parse_llm_response(response)
        
        assert result["verdict"] == "WARN"
        assert len(result["findings"]) == 1
    
    def test_parse_json_without_language_hint(self):
        """Test parsing JSON with generic markdown block."""
        response = '''```
{"verdict": "BLOCK", "findings": []}
```'''
        
        result = parse_llm_response(response)
        
        assert result["verdict"] == "BLOCK"
    
    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_llm_response("not valid json")
    
    def test_parse_empty_response(self):
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError):
            parse_llm_response("")


class TestValidateResponseStructure:
    """Tests for validate_response_structure function."""
    
    def test_valid_response(self):
        """Test validating a complete valid response."""
        data = {
            "verdict": "PASS",
            "findings": [
                {
                    "severity": "minor",
                    "category": "style",
                    "description": "Minor issue",
                    "line_number": 10,
                    "suggestion": "Fix it",
                }
            ],
            "summary": "OK",
        }
        
        result = validate_response_structure(data)
        
        assert result["verdict"] == "PASS"
        assert len(result["findings"]) == 1
    
    def test_missing_findings(self):
        """Test that missing findings defaults to empty list."""
        data = {"verdict": "PASS"}
        
        result = validate_response_structure(data)
        
        assert result["verdict"] == "PASS"
        assert result["findings"] == []
    
    def test_missing_verdict(self):
        """Test that missing verdict raises ValueError."""
        data = {"findings": []}
        
        with pytest.raises(ValueError, match="Missing required field: verdict"):
            validate_response_structure(data)
    
    def test_invalid_verdict_normalized(self):
        """Test that invalid verdict is normalized to WARN."""
        data = {"verdict": "UNKNOWN", "findings": []}
        
        result = validate_response_structure(data)
        
        assert result["verdict"] == "WARN"
    
    def test_invalid_severity_normalized(self):
        """Test that invalid severity is normalized to minor."""
        data = {
            "verdict": "PASS",
            "findings": [{"severity": "unknown", "category": "bug", "description": "x"}],
        }
        
        result = validate_response_structure(data)
        
        assert result["findings"][0]["severity"] == "minor"
    
    def test_invalid_category_normalized(self):
        """Test that invalid category is normalized."""
        data = {
            "verdict": "PASS",
            "findings": [{"severity": "minor", "category": "unknown", "description": "x"}],
        }
        
        result = validate_response_structure(data)
        
        assert result["findings"][0]["category"] == "maintainability"
    
    def test_case_insensitive_verdict(self):
        """Test that verdict is case-insensitive."""
        data = {"verdict": "block", "findings": []}
        
        result = validate_response_structure(data)
        
        assert result["verdict"] == "BLOCK"


class TestFindingsToJson:
    """Tests for findings_to_json function."""
    
    def test_empty_list(self):
        """Test converting empty findings list."""
        result = findings_to_json([])
        assert result == "[]"
    
    def test_single_finding(self):
        """Test converting single finding."""
        findings = [
            Finding(
                severity="critical",
                category="bug",
                description="Test bug",
                line_number=42,
                suggestion="Fix it",
                file_path="test.py",
            )
        ]
        
        result = findings_to_json(findings)
        parsed = json.loads(result)
        
        assert len(parsed) == 1
        assert parsed[0]["severity"] == "critical"
        assert parsed[0]["description"] == "Test bug"


class TestReviewPipeline:
    """Tests for ReviewPipeline class."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock(spec=LLMClient)
        client.model = "test-model"
        client.timeout = 100
        
        # Mock chat_completion
        async def mock_chat(*args, **kwargs):
            return ChatCompletionResponse(
                content=json.dumps({"verdict": "PASS", "findings": [], "summary": "OK"}),
                model="test-model",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                finish_reason="stop",
            )
        
        client.chat_completion = AsyncMock(side_effect=mock_chat)
        return client
    
    @pytest.fixture
    def sample_target(self):
        """Create a sample review target."""
        return ReviewTarget(
            files=[Path("test.py")],
            source_type=TargetSourceType.FILES,
        )
    
    def test_get_stages_for_light_mode(self):
        """Test stage selection for light mode."""
        pipeline = ReviewPipeline(
            llm_client=MagicMock(),
            mode=ReviewMode.LIGHT,
        )
        stages = pipeline._get_stages_for_mode()
        assert stages == ["precheck"]
    
    def test_get_stages_for_standard_mode(self):
        """Test stage selection for standard mode."""
        pipeline = ReviewPipeline(
            llm_client=MagicMock(),
            mode=ReviewMode.STANDARD,
        )
        stages = pipeline._get_stages_for_mode()
        assert stages == ["precheck", "diff_analysis"]
    
    def test_get_stages_for_deep_mode(self):
        """Test stage selection for deep mode."""
        pipeline = ReviewPipeline(
            llm_client=MagicMock(),
            mode=ReviewMode.DEEP,
        )
        stages = pipeline._get_stages_for_mode()
        assert stages == ["precheck", "diff_analysis", "context_read", "verify"]
    
    @pytest.mark.asyncio
    async def test_review_file_basic(self, mock_llm_client, sample_target, tmp_path):
        """Test reviewing a single file."""
        # Create temp file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('world')\n")
        
        pipeline = ReviewPipeline(llm_client=mock_llm_client)
        result = await pipeline.review_file(sample_target, test_file)
        
        assert result.overall_verdict == Verdict.PASS
        assert result.blocked is False
        assert len(result.stage_results) == 2  # standard mode has 2 stages
    
    @pytest.mark.asyncio
    async def test_review_file_fail_fast_on_critical(
        self, mock_llm_client, sample_target, tmp_path
    ):
        """Test fail-fast behavior when critical findings are found."""
        # Create temp file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test code")
        
        # Mock response with critical finding
        async def mock_chat_critical(*args, **kwargs):
            return ChatCompletionResponse(
                content=json.dumps({
                    "verdict": "BLOCK",
                    "findings": [{
                        "severity": "critical",
                        "category": "security",
                        "description": "Critical security issue",
                    }],
                    "summary": "Blocked",
                }),
                model="test-model",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                finish_reason="stop",
            )
        
        mock_llm_client.chat_completion = AsyncMock(side_effect=mock_chat_critical)
        
        pipeline = ReviewPipeline(
            llm_client=mock_llm_client,
            mode=ReviewMode.STANDARD,
            fail_fast_threshold=1,
        )
        
        result = await pipeline.review_file(sample_target, test_file)
        
        assert result.blocked is True
        assert result.blocked_at_stage == "precheck"
        assert result.overall_verdict == Verdict.BLOCK
        # Should stop after first stage due to fail-fast
        assert len(result.stage_results) == 1
    
    @pytest.mark.asyncio
    async def test_review_file_continues_without_critical(
        self, mock_llm_client, sample_target, tmp_path
    ):
        """Test that pipeline continues when no critical findings."""
        # Create temp file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test code")
        
        # Mock response with only minor findings
        async def mock_chat_minor(*args, **kwargs):
            return ChatCompletionResponse(
                content=json.dumps({
                    "verdict": "WARN",
                    "findings": [{
                        "severity": "minor",
                        "category": "style",
                        "description": "Minor style issue",
                    }],
                    "summary": "Warnings only",
                }),
                model="test-model",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                finish_reason="stop",
            )
        
        mock_llm_client.chat_completion = AsyncMock(side_effect=mock_chat_minor)
        
        pipeline = ReviewPipeline(
            llm_client=mock_llm_client,
            mode=ReviewMode.STANDARD,
            fail_fast_threshold=1,
        )
        
        result = await pipeline.review_file(sample_target, test_file)
        
        # Should run both stages (no fail-fast)
        assert len(result.stage_results) == 2
        # Only minor findings = PASS (not WARN, since WARN requires major findings)
        assert result.overall_verdict == Verdict.PASS


class TestRunReview:
    """Tests for run_review convenience function."""
    
    @pytest.mark.asyncio
    async def test_run_review_with_mode(self):
        """Test run_review with different modes."""
        target = ReviewTarget(
            files=[],
            source_type=TargetSourceType.FILES,
        )
        
        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "test"
        mock_client.timeout = 60
        mock_client.chat_completion = AsyncMock(return_value=ChatCompletionResponse(
            content='{"verdict": "PASS", "findings": []}',
            model="test",
            usage=TokenUsage(),
            finish_reason="stop",
        ))
        
        # Test light mode
        results = await run_review(target, mock_client, mode="light")
        assert isinstance(results, list)
        
        # Test standard mode
        results = await run_review(target, mock_client, mode="standard")
        assert isinstance(results, list)
        
        # Test deep mode
        results = await run_review(target, mock_client, mode="deep")
        assert isinstance(results, list)
        
        # Test invalid mode defaults to standard
        results = await run_review(target, mock_client, mode="invalid")
        assert isinstance(results, list)


class TestIntegration:
    """Integration tests for the pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_mocked_llm(self, tmp_path):
        """Test full pipeline flow with mocked LLM."""
        # Create test files
        file1 = tmp_path / "good.py"
        file1.write_text("def good_code():\n    return 42\n")
        
        file2 = tmp_path / "bad.py"
        file2.write_text("def bad_code():\n    # TODO: implement\n    pass\n")
        
        target = ReviewTarget(
            files=[file1, file2],
            source_type=TargetSourceType.FILES,
        )
        
        # Create mock client
        mock_client = MagicMock(spec=LLMClient)
        mock_client.model = "test-model"
        mock_client.timeout = 100
        
        call_count = 0
        
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Alternate between PASS and WARN responses
            if call_count % 3 == 0:
                content = json.dumps({
                    "verdict": "WARN",
                    "findings": [{
                        "severity": "minor",
                        "category": "style",
                        "description": "Consider adding docstring",
                    }],
                    "summary": "Minor issues",
                })
            else:
                content = json.dumps({
                    "verdict": "PASS",
                    "findings": [],
                    "summary": "Looks good",
                })
            
            return ChatCompletionResponse(
                content=content,
                model="test-model",
                usage=TokenUsage(input_tokens=20, output_tokens=10),
                finish_reason="stop",
            )
        
        mock_client.chat_completion = AsyncMock(side_effect=mock_chat)
        
        # Run pipeline
        pipeline = ReviewPipeline(
            llm_client=mock_client,
            mode=ReviewMode.STANDARD,
            fail_fast_threshold=1,
        )
        
        results = await pipeline.review(target)
        
        assert len(results) == 2
        assert all(isinstance(r, ReviewResult) for r in results)
        
        # Verify LLM was called
        assert call_count > 0
