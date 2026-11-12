# Claude Fable 5.2 runs (second LLM)

Claude Fable 5.2 (Anthropic, released Nov 2026) repeats conditions F (3 independent chats), G and H with
exactly the prompts in ../astra/ (PROMPT_F_diagnostics.md, PROMPT_G_resnet_head.md, PROMPT_H_edit_report.md),
the same attached files, practice slices, budget, rules and rivals as GPT-6 Astra. The prompts were fixed
before any Fable run and do not name either model.

Setup per run: new chat, outside any Project, memory and code execution off; transcripts checked for tool use.
Disclosure: the pipeline and prompts were drafted with Claude (Anthropic), the same family as Fable.

Files: practice_F<run>/, practice_log_F<run>.jsonl, head/, practice_log_G.jsonl, edit_h/, practice_log_H.jsonl,
transcripts transcript_*.md. Results: results/written__fable-diagnose-r<run>.json, head_resnet50_fable.*,
edit_resnet50_H_fable.*. Notebook cells FB-F2/F3, FB-G2/G3, FB-H2/H3 (setup cells F1, G1, H1 are shared).
