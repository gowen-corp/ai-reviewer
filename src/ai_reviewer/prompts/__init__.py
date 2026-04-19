"""Prompts for AI code review stages.

This module contains all prompt templates used in the review pipeline.
Each prompt is designed to elicit structured JSON responses from the LLM.

Response format for all prompts:
{
    "verdict": "PASS|BLOCK|WARN",
    "findings": [
        {
            "severity": "critical|major|minor",
            "category": "bug|security|performance|style|maintainability",
            "description": "...",
            "line_number": null,  // optional
            "suggestion": "..."
        }
    ],
    "summary": "..."  // optional brief summary
}
"""

from __future__ import annotations

# System prompt for all review stages
SYSTEM_PROMPT = """You are an expert code reviewer. Your task is to analyze code and provide structured feedback.

Always respond with valid JSON in this exact format:
{
    "verdict": "PASS|BLOCK|WARN",
    "findings": [
        {
            "severity": "critical|major|minor",
            "category": "bug|security|performance|style|maintainability",
            "description": "Clear description of the issue",
            "line_number": 42,
            "suggestion": "How to fix it"
        }
    ],
    "summary": "Brief overall assessment"
}

Verdict meanings:
- PASS: Code meets quality standards, no significant issues
- WARN: Code has issues but they are not blocking (warnings/minor findings only)
- BLOCK: Code has critical/major issues that must be fixed before merging

Severity levels:
- critical: Security vulnerabilities, data loss, crashes, major bugs
- major: Logic errors, performance issues, significant maintainability problems
- minor: Style issues, suggestions, minor improvements

Categories:
- bug: Functional bugs, logic errors
- security: Security vulnerabilities, exposed secrets
- performance: Performance issues, inefficiencies
- style: Code style, formatting, naming
- maintainability: Code structure, documentation, complexity
"""

# Stage 1: PRECHECK - Quick syntax and basic quality check
PRECHECK_PROMPT = """## Task: Pre-check Code Quality

Perform a quick analysis of the following code to identify obvious issues:
- Syntax errors or malformed code
- Merge conflict markers (<<<<<<, ======, >>>>>>)
- Obvious bugs or typos
- Missing imports or undefined variables
- Basic code structure issues

## Code to Review:

```{language}
{code_content}
```

## Instructions:
1. Scan for syntax errors and structural issues
2. Look for merge conflict markers
3. Identify any obvious bugs
4. If you find critical issues, set verdict to BLOCK

Respond with your findings in JSON format.
"""

# Stage 2: DIFF_ANALYSIS - Analyze changes in context
DIFF_ANALYSIS_PROMPT = """## Task: Analyze Code Changes

Review the following diff to understand what changed and assess the quality of changes.

## File: {file_path}

### Original Code:
```{language}
{original_code}
```

### New Code:
```{language}
{new_code}
```

### Diff Context:
```diff
{diff_content}
```

## Instructions:
1. Understand what functionality was added/modified/removed
2. Assess if the changes introduce bugs or regressions
3. Check if error handling is adequate
4. Verify that the changes follow best practices
5. Look for potential side effects

Focus on:
- Logic correctness
- Error handling
- Edge cases
- Backward compatibility

Respond with your findings in JSON format.
"""

# Stage 3: CONTEXT_READ - Deep analysis with broader context
CONTEXT_READ_PROMPT = """## Task: Deep Code Analysis with Context

Perform a thorough analysis of the code considering its role in the larger system.

## File: {file_path}

### Full File Content:
```{language}
{full_code}
```

### Change Summary:
{change_summary}

### Related Findings from Previous Stages:
{previous_findings}

## Instructions:
1. Analyze code architecture and design patterns
2. Check for proper abstraction and separation of concerns
3. Evaluate testability of the code
4. Assess documentation and comments quality
5. Look for code smells and anti-patterns
6. Consider security implications
7. Evaluate performance characteristics

Be thorough but practical. Focus on issues that impact:
- Correctness
- Security
- Maintainability
- Performance

Respond with your findings in JSON format.
"""

# Stage 4: VERIFY - Final verification and aggregation
VERIFY_PROMPT = """## Task: Final Verification and Synthesis

Synthesize all findings from previous review stages and provide a final assessment.

## File: {file_path}

### All Findings So Far:
{all_findings_json}

### Review Mode: {review_mode}

## Instructions:
1. Review all findings from previous stages
2. Deduplicate similar findings
3. Prioritize by severity
4. Make a final verdict decision
5. Ensure each finding has a clear suggestion

Consider the review mode:
- light: Focus only on critical blockers
- standard: Balance between bugs and quality
- deep: Comprehensive analysis including style and maintainability

Final verdict criteria:
- BLOCK: Any critical findings OR multiple major findings
- WARN: Only minor/major findings that don't block merging
- PASS: No significant issues found

Respond with your final assessment in JSON format.
"""

# Aggregation prompt for multi-file reviews
AGGREGATION_PROMPT = """## Task: Aggregate Multi-File Review Results

Synthesize findings from multiple files into a cohesive review report.

## Files Reviewed:
{file_list}

## Per-File Findings:
{per_file_findings}

## Instructions:
1. Combine findings from all files
2. Identify cross-file patterns or systemic issues
3. Prioritize findings by severity across all files
4. Provide an overall project-level verdict
5. Highlight the most important issues to address first

Overall verdict criteria:
- BLOCK: Any file has critical issues OR multiple files have major issues
- WARN: Issues exist but are not blocking
- PASS: All files meet quality standards

Respond with aggregated findings in JSON format.
"""


def get_prompt_for_stage(stage: str, **kwargs) -> str:
    """Get the prompt template for a specific review stage.
    
    Args:
        stage: Stage name (precheck, diff_analysis, context_read, verify, aggregation).
        **kwargs: Template variables to substitute.
        
    Returns:
        Formatted prompt string.
        
    Raises:
        ValueError: If stage name is unknown.
    """
    prompts = {
        "precheck": PRECHECK_PROMPT,
        "diff_analysis": DIFF_ANALYSIS_PROMPT,
        "context_read": CONTEXT_READ_PROMPT,
        "verify": VERIFY_PROMPT,
        "aggregation": AGGREGATION_PROMPT,
    }
    
    if stage not in prompts:
        raise ValueError(f"Unknown stage: {stage}. Valid stages: {list(prompts.keys())}")
    
    template = prompts[stage]
    
    # Simple string substitution for template variables
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    return result
