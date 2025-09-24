const { execSync } = require('child_process');

const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m',
    red: '\x1b[31m'
};

function getChangedFiles() {
    try {
        const status = execSync('git status --porcelain', { encoding: 'utf8' });
        const files = {
            added: [],
            modified: [],
            deleted: []
        };
        
        status.split('\n').filter(line => line.trim()).forEach(line => {
            const statusCode = line.substring(0, 2);
            const file = line.substring(3).trim();
            
            if (!file || file.startsWith('.git/')) return;
            
            if (statusCode.includes('D')) {
                files.deleted.push(file);
            } else if (statusCode.includes('A') || statusCode === '??') {
                files.added.push(file);
            } else {
                files.modified.push(file);
            }
        });
        
        return files;
    } catch (error) {
        return { added: [], modified: [], deleted: [] };
    }
}

function isFirstCommit(file) {
    try {
        execSync(`git log --oneline "${file}"`, { stdio: 'pipe' });
        return false; // File has commit history
    } catch (error) {
        return true; // File not in git history yet
    }
}

async function autoCommit() {
    console.log(`${colors.cyan}🚀 Quick Auto-Commit${colors.reset}\n`);
    
    const files = getChangedFiles();
    const totalFiles = files.added.length + files.modified.length + files.deleted.length;
    
    if (totalFiles === 0) {
        console.log(`${colors.yellow}No changes to commit.${colors.reset}`);
        return;
    }
    
    console.log(`${colors.blue}Found ${totalFiles} changed files${colors.reset}`);
    if (files.added.length > 0) console.log(`${colors.green}  ${files.added.length} added${colors.reset}`);
    if (files.modified.length > 0) console.log(`${colors.yellow}  ${files.modified.length} modified${colors.reset}`);
    if (files.deleted.length > 0) console.log(`${colors.red}  ${files.deleted.length} deleted${colors.reset}`);
    
    // Add all files (including deletions)
    try {
        execSync('git add .', { stdio: 'pipe' });
        console.log(`${colors.green}✓ Added all changes${colors.reset}`);
    } catch (error) {
        console.log(`${colors.red}✗ Failed to add changes${colors.reset}`);
        return;
    }
    
    // Create commit message based on file changes
    let commitMessage;
    if (totalFiles === 1) {
        let file, action;
        if (files.deleted.length === 1) {
            file = files.deleted[0];
            action = 'remove';
        } else if (files.added.length === 1) {
            file = files.added[0];
            action = 'upload';
        } else {
            file = files.modified[0];
            action = 'update';
        }
        commitMessage = `${action} ${file}`;
    } else {
        // Multiple files - prioritize the most significant action
        if (files.deleted.length > 0) {
            if (files.deleted.length === totalFiles) {
                commitMessage = `remove ${files.deleted.length} files`;
            } else {
                commitMessage = `update ${totalFiles} files (${files.deleted.length} removed)`;
            }
        } else if (files.added.length > 0) {
            commitMessage = `upload ${totalFiles} files`;
        } else {
            commitMessage = `update ${totalFiles} files`;
        }
    }
    
    // Commit
    try {
        execSync(`git commit -m "${commitMessage}"`, { stdio: 'pipe' });
        console.log(`${colors.green}✓ Committed: "${commitMessage}"${colors.reset}`);
    } catch (error) {
        console.log(`${colors.red}✗ Failed to commit${colors.reset}`);
        return;
    }
    
    // Push
    try {
        execSync('git push', { stdio: 'pipe' });
        console.log(`${colors.green}✓ Pushed to remote${colors.reset}`);
    } catch (error) {
        console.log(`${colors.red}✗ Failed to push${colors.reset}`);
        return;
    }
    
    console.log(`\n${colors.cyan}🎉 Done!${colors.reset}`);
}

process.on('SIGINT', () => {
    console.log(`\n${colors.yellow}Interrupted${colors.reset}`);
    process.exit(0);
});

autoCommit().catch(error => {
    console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
    process.exit(1);
});
