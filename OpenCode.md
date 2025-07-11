# OpenCode.md

## Build Instructions

To build the project, use the following command:

```bash
cd src && agda -c Everything.agda
```

## Linting

No dedicated linting command found. Code correctness is enforced by Agda's type system during compilation.

## Testing

To run tests, use the following command:

```bash
nix flake check
```

## Code Style

- Adhere to existing Agda code conventions within the `src/` directory.
- Use descriptive names for modules, definitions, and variables.
- Keep lines concise, ideally under 80 characters.
- Module structure should mirror the directory hierarchy in `src/`.
- Use explicit and qualified imports to prevent naming conflicts.
- All top-level definitions should have type signatures.
