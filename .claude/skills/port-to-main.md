# Port Changes to Main Repository

This skill ports changes from the AIE9 Staging repository to the main AIE9 repository.

## Repository Paths
- **Staging**: `/home/chris/Code/AI Makerspace/Classes/AIE9 - Staging`
- **Main**: `/home/chris/Code/AI Makerspace/Classes/AIE9`

## Instructions

When this skill is invoked, follow these steps:

### 1. Compare Repositories
First, compare the staging and main repositories to identify differences:

```bash
diff -rq "/home/chris/Code/AI Makerspace/Classes/AIE9 - Staging" "/home/chris/Code/AI Makerspace/Classes/AIE9" --exclude='.git' --exclude='.claude'
```

### 2. Show Changes Summary
Present a clear summary of:
- Files that exist only in staging (new files to port)
- Files that exist only in main (files that may have been removed from staging)
- Files that differ between the two repositories

### 3. Ask User Which Changes to Port
Use AskUserQuestion to confirm which changes the user wants to port:
- All changes
- Select specific files/directories
- Skip certain files

### 4. Copy Changes
For approved changes, use rsync to copy from staging to main:

```bash
rsync -av --exclude='.git' --exclude='.claude' "<source_path>" "<dest_path>"
```

### 5. Review and Commit (Optional)
After copying:
1. Show `git status` in the main repository
2. Ask if the user wants to commit the changes
3. If yes, create a commit with an appropriate message describing what was ported

## Notes
- Always exclude `.git` and `.claude` directories when comparing/copying
- The staging repo may contain work-in-progress that shouldn't be ported yet
- Always confirm before overwriting files in main
