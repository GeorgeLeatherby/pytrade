# COPILOT EDITS OPERATIONAL GUIDELINES

## PRIME DIRECTIVE
Avoid working on more then one file at a time. Multiple simulataneos edits to a file will cause corruption. 

Always explain and teach about what you are doing while coding.

When working with large files (> 100 lines) or complex changes, always start by creating a detailed plan BEFORE making any edits. If you discover additional needed changes during editing, update the plan and review the new plan before continuing.

Your plan MUST include:
  - All functions/sections that need modification and the reason why they need modification
  - The order in which changes should be applied
  - Dependencies between changes
  - Estimated number of seperate edits required

ALWAYS add the line number and the filename when you reference code.

Always verify information before presenting it. Do not make assumptions or speculate without clear evidence.

Whenever uncertainties arise, ask for clarification before proceeding. Do not spend tokens and time trying to guess user intent. Always trigger user confirmation.

## Coding style guidelines
Use descriptive variable and function names that clearly indicate their purpose. Check if there are existing naming conventions in the codebase and follow them for consistency. 
Check for existing functions that perform similar tasks and reuse their naming patterns when creating new functions. Avoid using abbreviations or acronyms unless they are widely recognized and commonly used in the codebase.
Write modular code by breaking down complex functions into smaller, reusable functions. Apply strict seperation of concerns.
Do NOT introduce fallbacks, default values, or error handling unless explicitly requested. Fail-first approach.
Reuse existing functions and code patterns whenever possible to maintain consistency across the codebase.
When making changes, ensure that the code remains clean and well-organized.
Do NOT remove unrelated code or functionality; preserve existing structures.
If a change has been applied and verified, remove unused code or comments that are no longer relevant. ALWAYS clearly inform the user about deleted code and the reason for its removal.
Provide professional, clear and concise comments explaining the purpose of code sections, but avoid over-commenting. Comments should add value and not state the obvious.