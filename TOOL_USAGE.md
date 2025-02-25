# Tools Implementation Guide

This document outlines the tools available in the code-vectorizer system and provides implementation guidance for each tool.

## Overview

The system defines four core tools that enable file system operations and command execution:
- `list_dir`: Directory listing
- `read_file`: File content reading
- `edit_file`: File modification
- `run_terminal_cmd`: Command execution

Each tool is defined with a specific parameter schema and must be implemented according to these specifications.

## Tool Specifications

### 1. list_dir

**Purpose**: Lists directory contents at a specified path relative to the workspace.

**Parameters**:
```typescript
{
  relative_workspace_path: string  // Path to list contents of
}
```

**Implementation Requirements**:
- Should accept a relative path from the workspace root
- Must return directory contents including both files and folders
- Should handle both shallow and nested paths
- Must handle errors for invalid/non-existent paths

**Example Implementation**:
```typescript
async function listDir(params: { relative_workspace_path: string }) {
  const fullPath = path.join(workingDir, params.relative_workspace_path);
  
  try {
    const contents = await fs.readdir(fullPath);
    return {
      success: true,
      contents: contents
    };
  } catch (error) {
    return {
      success: false,
      error: `Failed to list directory: ${error.message}`
    };
  }
}
```

### 2. read_file

**Purpose**: Reads file contents, with support for both full file and partial (line-based) reading.

**Parameters**:
```typescript
{
  relative_workspace_path: string,    // Path to the file
  should_read_entire_file: boolean,   // Whether to read entire file
  start_line_one_indexed?: number,    // Optional: Start line (1-based)
  end_line_one_indexed_inclusive?: number  // Optional: End line (1-based)
}
```

**Implementation Requirements**:
- Must support reading entire files when `should_read_entire_file` is true
- Must support reading specific line ranges when provided
- Should validate line numbers are within file bounds
- Must handle various file encodings (UTF-8 recommended default)
- Should handle errors for invalid files/permissions

**Example Implementation**:
```typescript
async function readFile(params: {
  relative_workspace_path: string,
  should_read_entire_file: boolean,
  start_line_one_indexed?: number,
  end_line_one_indexed_inclusive?: number
}) {
  const fullPath = path.join(workingDir, params.relative_workspace_path);
  
  try {
    if (params.should_read_entire_file) {
      const content = await fs.readFile(fullPath, 'utf8');
      return { success: true, content };
    }

    const content = await fs.readFile(fullPath, 'utf8');
    const lines = content.split('\n');
    const start = params.start_line_one_indexed - 1;
    const end = params.end_line_one_indexed_inclusive;
    
    return {
      success: true,
      content: lines.slice(start, end).join('\n')
    };
  } catch (error) {
    return {
      success: false,
      error: `Failed to read file: ${error.message}`
    };
  }
}
```

### 3. edit_file

**Purpose**: Makes code changes to specified files based on instructions.

**Parameters**:
```typescript
{
  target_file: string,     // File to edit
  instructions: string,    // What changes to make
  code_edit: string       // The actual edit content
}
```

**Implementation Requirements**:
- Must validate target file exists before attempting edits
- Should create backup of file before editing
- Must handle partial file updates
- Should implement error recovery in case of failed edits
- Must maintain file permissions
- Should validate edit instructions against actual changes

**Example Implementation**:
```typescript
async function editFile(params: {
  target_file: string,
  instructions: string,
  code_edit: string
}) {
  const fullPath = path.join(workingDir, params.target_file);
  
  try {
    // Create backup
    await fs.copyFile(fullPath, `${fullPath}.bak`);
    
    // Perform edit
    await fs.writeFile(fullPath, params.code_edit, 'utf8');
    
    return {
      success: true,
      message: `File ${params.target_file} updated successfully`
    };
  } catch (error) {
    // Restore from backup if exists
    try {
      await fs.copyFile(`${fullPath}.bak`, fullPath);
    } catch (backupError) {
      // Handle backup restoration failure
    }
    
    return {
      success: false,
      error: `Failed to edit file: ${error.message}`
    };
  }
}
```

### 4. run_terminal_cmd

**Purpose**: Executes terminal commands with configurable execution options.

**Parameters**:
```typescript
{
  command: string,              // Command to execute
  is_background: boolean,       // Whether to run in background
  require_user_approval: boolean // Whether user must approve
}
```

**Implementation Requirements**:
- Must support both foreground and background execution
- Should implement user approval mechanism when required
- Must handle command timeout scenarios
- Should capture and return both stdout and stderr
- Must implement proper process cleanup for background tasks
- Should handle shell-specific command formatting

**Example Implementation**:
```typescript
async function runTerminalCmd(params: {
  command: string,
  is_background: boolean,
  require_user_approval: boolean
}) {
  try {
    if (params.require_user_approval) {
      const approved = await getUserApproval(params.command);
      if (!approved) {
        return {
          success: false,
          error: 'Command execution rejected by user'
        };
      }
    }

    if (params.is_background) {
      const process = spawn(params.command, [], {
        shell: true,
        detached: true,
        stdio: 'ignore'
      });
      process.unref();
      return {
        success: true,
        pid: process.pid
      };
    }

    const { stdout, stderr } = await exec(params.command);
    return {
      success: true,
      stdout,
      stderr
    };
  } catch (error) {
    return {
      success: false,
      error: `Command execution failed: ${error.message}`
    };
  }
}
```

## Integration Notes

1. All tools should be implemented as async functions
2. Error handling should be consistent across all tools
3. Tools should validate input parameters before execution
4. Each tool should return a consistent response format:
   ```typescript
   {
     success: boolean,
     error?: string,
     [key: string]: any  // Additional tool-specific response data
   }
   ```
5. Tools should respect workspace boundaries and not access files outside the working directory

## Security Considerations

1. Validate all file paths to prevent directory traversal attacks
2. Implement proper permission checks before file operations
3. Sanitize command inputs for `run_terminal_cmd`
4. Implement rate limiting for resource-intensive operations
5. Consider implementing a timeout mechanism for long-running operations

## Tool Registration

Based on the implementation in `code-vectorizer/src/tools/index.ts`, tools are registered using a TypeScript interface that defines their schema and parameters. Here's how the tools are initialized:

```typescript
const tools: Tool[] = [
  {
    name: 'list_dir',
    description: 'List directory contents',
    parameters: {
      required: ['relative_workspace_path'],
      properties: {
        relative_workspace_path: {
          type: 'string',
          description: 'Path to list contents of'
        }
      }
    }
  },
  // ... other tool definitions
];

const implementations = createToolImplementations(workingDir);
const executor = new AgentExecutor(tools, implementations);
```

This registration system ensures type safety and parameter validation at runtime.

## Conclusion

This documentation provides a solid foundation for implementing these tools in a secure and reliable manner. The system is designed to be extensible, allowing for additional tools to be added while maintaining a consistent interface and error handling approach. 