# TH105 autoplay working rules

- Do not use REA, any REA-provided tool, or LeanToken MCP.
- The supported executable is `th105c.exe` with SHA-256
  `49c23d9467b9927ba687ed2b873c4bc2d2f39ddadc9f55051ccf10172c0b7c11`.
- Physical input uses foreground-guarded Win32 `SendInput` scan codes. Release every
  injected key on normal exit, error, and Ctrl-C.
- Read-only process-memory sensing is allowed. Keep runtime writes/patches separate
  from the ordinary launcher and require explicit opt-in for them.
- Menu automation must validate native scene transitions; screenshots are only
  bootstrap evidence.
- Never stop an ambiguous process name. Resolve and verify the exact executable
  path and PID first.
