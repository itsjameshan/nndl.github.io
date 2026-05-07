---
name: reformat-exam-docx-template
description: Reformat Chinese exam paper DOCX files into a Word exam template. Use when Codex needs to rebuild a paper such as 试卷1.docx with a supplied 试卷模板.docx/.doc style, preserve questions and scores, create a new *_新模板.docx output, fix header/page-frame/table layout issues, and verify DOCX structure and question counts.
---

# Reformat Exam DOCX Template

Use this skill to rebuild an existing exam paper into a supplied Word exam template while preserving the original paper and answer files.

## Inputs

- Source paper: a `.docx` such as `answers_fixed_utf8_试卷1-10全部/试卷1.docx`.
- Template: prefer `试卷模板.docx`; if only `试卷模板.doc` exists, convert it to temporary `.docx` first.
- Output: write a new file beside the source, named like `试卷1_新模板.docx`.
- Do not modify answer files such as `试卷1_答案 改.docx`.

## Workflow

1. Inspect the source, template, and output paths. Do not overwrite the original paper.
2. Extract template OOXML with `zipfile` and `lxml`; read real font, size, table, header, and border settings instead of guessing from screenshots.
3. Parse the source paper by section headings:
   - 单项选择题: 20 items, 30 points.
   - 填空题: 10 items, 10 points.
   - 判断题: 10 items, 10 points.
   - 多项选择题: 10 items, 20 points.
   - 简答题/论述题: 2 items, 30 points.
4. Build the output from the editable template with `python-docx`, then clear the body and reinsert content using real Word objects.
5. Fill header metadata only from explicit user instructions, the source paper, or the template. Do not hard-code course-specific metadata in this skill.
6. Preserve the score distribution and total score of 100.

## Layout Rules

- Keep the page outer frame as a repeated header VML rectangle. Do not replace it with plain text or ASCII lines.
- If a horizontal line appears above `昆明学院`, check the template `header` paragraph style first. Remove its `w:pBdr/w:bottom`; do not remove the outer frame top border.
- Use true Word tables for the main score table, per-section score boxes, and objective answer grids.
- Put each section title on the left and the `得分` box on the right. The score box must have visible vertical borders.
- Widen objective answer grids so they span the usable page width naturally; avoid leaving them squeezed in the center.
- Use simple page numbers (`PAGE` only), not `第 X 页 共 Y 页`, unless the user asks otherwise.
- Keep course metadata generic. Course title, course code, exam time, target class, exam mode, instructor, and department-chair fields must be user-supplied, source-derived, or left blank.
- Use the template typography where practical:
  - Main title: `隶书`, 18 pt.
  - Header fields and notices: about 14 pt.
  - Score table labels: `隶书`, 12 pt.
  - Section headings: about 15 pt, bold.
  - Actual question and option text: 12 pt when matching the final accepted version.
- Rename the final section from `简答题` to `论述题` when the user requests the accepted format used for this project.

## Implementation Notes

- Use `python-docx` for document construction and `lxml` for OOXML checks.
- Use invisible white filler text only when needed to stabilize blank table cell widths; keep it out of numbered cells so extracted numbers remain clean.
- For underlined blanks after labels, add underlined runs containing the known value plus spaces.
- Remove inherited template body content before rebuilding, but preserve useful template styles.
- If the template contains a header style border, remove it from `styles.xml` through the document style object before saving.
- Keep generated scripts in a temp or project helper path, but make the final deliverable the `.docx`.

## Verification

Run these checks before reporting completion:

- `unzip -t <output.docx>` succeeds.
- Extract output text and confirm counts: 20 single choice, 10 fill, 10 judgement, 10 multiple choice, 2 discussion/essay questions.
- Confirm score table totals 100.
- Confirm the output contains `五、论述题` when that wording was requested, and does not still show `五、简答题`.
- Inspect `word/header1.xml` and `word/styles.xml`:
  - The outer frame exists as `ExamPageFrame`.
  - The header paragraph/style has no `w:pBdr/w:bottom` stray line.
- Confirm actual question runs use 12 pt (`w:sz` value `24`) when required.
- Try the Documents skill renderer (`render_docx.py`) for visual QA. If `soffice/LibreOffice` is unavailable, state that visual rendering could not be completed and report the structural checks instead.

## Common Pitfalls

- Do not confuse the page frame top border with the template header style bottom border. The screenshot line above `昆明学院` is usually the header style border.
- Do not simulate answer grids or score boxes with text spacing; use real Word tables.
- Do not invent missing course title, course number, exam time, class, exam form, teacher, or department-chair values.
- Do not change answer documents unless explicitly asked.
- Do not commit or overwrite previous versions unless the user asks.
