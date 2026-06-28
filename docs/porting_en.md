Porting Guide

Supported Platforms

Platform   Status   Notes
Host       ✅ Full  Development and debugging
ARM        ✅ Full  Pure-C version
RISC-V     ✅ Full  Pure-C version

Adding a New Target

1. Create target directory under ops/
2. Add linker script link_xxx.ld
3. Add startup file start.S
4. Add build script build_xxx_*.sh

Adding a New Backend

1. Implement operator library (see ops/c/)
2. Modify codegen.py to add backend selection
3. Modify copy_files_to_build() to copy files

Accelerator Library Adaptation

1. Create accelerator directory under ops/xxx/
2. Implement accelerator operator wrappers
3. Modify build_xxx_*.sh to link accelerator library

Development Guidelines

- All commits must be in English
- All code comments must be in English
- Strict 80-column limit
- Follow existing code style
- Issues and PRs: English only
- Emails from Chinese users: Chinese is OK

Testing

python -m pytest tests/
