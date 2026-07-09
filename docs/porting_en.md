# Porting Guide

## Supported Platforms

| Platform | Status | Notes |
| :--- | :--- | :--- |
| Host | ✅ Full | Development and debugging |
| ARM | ✅ Full | Pure-C version |
| RISC-V | ✅ Full | Pure-C version |

## Adding a New Target

- Create target directory under ops/
- Add linker script link_xxx.ld 
- Add startup file start.S 
- Add build script build_xxx_*.sh

## Adding a New Backend

- Implement operator library (see ops/c/)
- Modify codegen.py to add backend selection 
- Modify copy_files_to_build() to copy files

## Accelerator Library Adaptation

- Create accelerator directory under ops/xxx/ 
- Implement accelerator operator wrappers 
- Modify build_xxx_*.sh to link accelerator library

## Development Guidelines

- All commits must be in English
- All code comments must be in English
- **Strict 80-column limit**
- Follow existing code style
- Issues and PRs: English only
- Emails from Chinese users: Chinese please.

## Testing

`python -m pytest tests/`
