const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Cool console colors and effects
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    bgBlue: '\x1b[44m',
    bgGreen: '\x1b[42m',
    bgYellow: '\x1b[43m',
    bgRed: '\x1b[41m',
    bgMagenta: '\x1b[45m',
    bgCyan: '\x1b[46m'
};

// Rainbow colors array
const rainbowColors = [colors.red, colors.yellow, colors.green, colors.cyan, colors.blue, colors.magenta];

// Animated text functions
function typeWriter(text, color = colors.white, delay = 50) {
    return new Promise((resolve) => {
        let i = 0;
        const timer = setInterval(() => {
            process.stdout.write(color + text[i] + colors.reset);
            i++;
            if (i >= text.length) {
                clearInterval(timer);
                process.stdout.write('\n');
                resolve();
            }
        }, delay);
    });
}

function rainbowText(text, speed = 100) {
    return new Promise((resolve) => {
        let colorIndex = 0;
        const cycles = 3;
        let currentCycle = 0;
        
        const timer = setInterval(() => {
            process.stdout.write('\r');
            for (let i = 0; i < text.length; i++) {
                const color = rainbowColors[(colorIndex + i) % rainbowColors.length];
                process.stdout.write(color + colors.bright + text[i] + colors.reset);
            }
            colorIndex = (colorIndex + 1) % rainbowColors.length;
            
            if (colorIndex === 0) {
                currentCycle++;
                if (currentCycle >= cycles) {
                    clearInterval(timer);
                    process.stdout.write('\n');
                    resolve();
                }
            }
        }, speed);
    });
}

function wavyText(text, color = colors.cyan) {
    return new Promise((resolve) => {
        const waves = ['~', '∿', '≈', '∼', '〰'];
        let waveIndex = 0;
        const cycles = 20;
        let currentCycle = 0;
        
        const timer = setInterval(() => {
            const wave = waves[waveIndex % waves.length];
            process.stdout.write(`\r${color}${colors.bright}${wave} ${text} ${wave}${colors.reset}`);
            waveIndex++;
            currentCycle++;
            
            if (currentCycle >= cycles) {
                clearInterval(timer);
                process.stdout.write('\n');
                resolve();
            }
        }, 150);
    });
}

function glowPulse(text, color = colors.cyan) {
    return new Promise((resolve) => {
        const intensities = [colors.dim, '', colors.bright];
        let index = 0;
        const cycles = 6;
        let currentCycle = 0;
        
        const timer = setInterval(() => {
            const intensity = intensities[index % intensities.length];
            process.stdout.write(`\r${color}${intensity}>>> ${text} <<<${colors.reset}`);
            index++;
            
            if (index % intensities.length === 0) {
                currentCycle++;
                if (currentCycle >= cycles) {
                    clearInterval(timer);
                    process.stdout.write('\n');
                    resolve();
                }
            }
        }, 200);
    });
}

function progressBar(current, total, message = '', width = 50) {
    const percentage = Math.round((current / total) * 100);
    const filledWidth = Math.round((current / total) * width);
    const emptyWidth = width - filledWidth;
    
    // Rainbow progress bar
    let filled = '';
    for (let i = 0; i < filledWidth; i++) {
        const color = rainbowColors[i % rainbowColors.length];
        filled += color + '█' + colors.reset;
    }
    const empty = colors.dim + '░'.repeat(emptyWidth) + colors.reset;
    
    const bar = `[${filled}${empty}]`;
    const percent = `${colors.bright}${colors.yellow}${percentage}%${colors.reset}`;
    
    process.stdout.write(`\r${bar} ${percent} ${colors.white}${message}${colors.reset}`);
    
    if (current === total) {
        process.stdout.write('\n');
    }
}

// Commit messages for different types of changes (no emojis)
const commitMessages = [
    "Initial quantum tunneling simulation setup",
    "Add QuTiP integration for quantum state evolution", 
    "Implement Crank-Nicolson time evolution scheme",
    "Add wave packet initialization and normalization",
    "Implement expectation value calculations",
    "Add transmission and reflection probability tracking",
    "Enhance uncertainty principle validation", 
    "Add rich console output formatting",
    "Implement data export for Manim visualization",
    "Create quantum tunneling animation with Manim",
    "Add absorbing boundary conditions",
    "Optimize absorber strength and width",
    "Improve animation visual elements",
    "Fix boundary reflection issues",
    "Polish simulation parameters and output",
    "Add comprehensive analysis plots",
    "Enhance animation with probability displays",
    "Fine-tune quantum mechanical accuracy",
    "Improve visual styling and layout",
    "Final optimization and cleanup",
    "Update documentation and comments",
    "Refactor code structure",
    "Performance optimizations",
    "Bug fixes and improvements",
    "Code cleanup and formatting"
];

// ASCII Art Banner
function showBanner() {
    const banner = `
${colors.cyan}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      QUANTUM TUNNELING AUTO-COMMIT WIZARD                    ║
║                                                              ║
║        Automatically committing your quantum code...         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝${colors.reset}
    `;
    console.log(banner);
}

// Get list of changed files (individual files, not folders)
function getChangedFiles() {
    try {
        const status = execSync('git status --porcelain', { encoding: 'utf8' });
        const files = status
            .split('\n')
            .filter(line => line.trim())
            .map(line => line.substring(3).trim())
            .filter(file => {
                // Include all files except git internal files
                if (!file || file.startsWith('.git/')) return false;
                
                // Include all file types: .py, .js, .json, .png, .md, etc.
                return true;
            });
        return files;
    } catch (error) {
        return [];
    }
}

// Execute git command with error handling
function gitCommand(command, description) {
    try {
        execSync(command, { stdio: 'pipe' });
        return true;
    } catch (error) {
        console.log(`${colors.red}>>> Failed: ${description}${colors.reset}`);
        console.log(`${colors.dim}Error: ${error.message}${colors.reset}`);
        return false;
    }
}

// Main auto-commit function
async function autoCommit() {
    showBanner();
    
    await rainbowText("QUANTUM TUNNELING AUTO-COMMIT SYSTEM", 80);
    await typeWriter("Scanning repository for changes...", colors.yellow, 30);
    
    const files = getChangedFiles();
    
    if (files.length === 0) {
        await glowPulse("Repository is clean! No changes to commit.", colors.green);
        return;
    }
    
    console.log(`${colors.bright}${colors.white}Found ${files.length} changed files:${colors.reset}`);
    files.forEach((file, index) => {
        console.log(`${colors.cyan}  ${index + 1}. ${file}${colors.reset}`);
    });
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    await wavyText("Starting auto-commit sequence...", colors.magenta);
    
    // Commit files one by one
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const message = commitMessages[i % commitMessages.length];
        
        await typeWriter(`\nPreparing commit ${i + 1}/${files.length}: ${file}`, colors.cyan, 20);
        
        // Progress animation with rainbow colors
        for (let j = 0; j <= 30; j++) {
            progressBar(j, 30, `Processing ${file}...`);
            await new Promise(resolve => setTimeout(resolve, 40));
        }
        
        // Add file
        if (gitCommand(`git add "${file}"`, `adding ${file}`)) {
            console.log(`${colors.green}>>> Added: ${file}${colors.reset}`);
        } else {
            continue;
        }
        
        // Commit file
        if (gitCommand(`git commit -m "${message}"`, `committing ${file}`)) {
            await glowPulse(`Committed: "${message}"`, colors.green);
        }
        
        // Cool delay between commits
        if (i < files.length - 1) {
            await wavyText("Preparing next commit...", colors.yellow);
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    
    // Final animation
    console.log('\n');
    for (let i = 0; i <= 100; i += 4) {
        progressBar(i, 100, "Finalizing quantum commit sequence...");
        await new Promise(resolve => setTimeout(resolve, 60));
    }
    
    await rainbowText("AUTO-COMMIT SEQUENCE COMPLETE!", 100);
    
    await typeWriter("Ready to push to repository...", colors.cyan, 40);
    
    // Ask if user wants to push
    const readline = require('readline');
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    rl.question(`${colors.yellow}Push commits to remote repository? (y/n): ${colors.reset}`, (answer) => {
        if (answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes') {
            console.log(`${colors.cyan}Pushing to remote...${colors.reset}`);
            if (gitCommand('git push', 'pushing to remote')) {
                glowPulse("Successfully pushed to remote repository!", colors.green);
            }
        } else {
            console.log(`${colors.yellow}Commits created locally. Use 'git push' when ready.${colors.reset}`);
        }
        rl.close();
        
        // Final celebration
        setTimeout(async () => {
            await rainbowText("QUANTUM TUNNELING PROJECT COMMITTED!", 120);
        }, 1000);
    });
}

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
    console.log(`\n\n${colors.red}>>> Auto-commit interrupted by user${colors.reset}`);
    process.exit(0);
});

// Run the auto-commit wizard
autoCommit().catch(error => {
    console.error(`${colors.red}>>> Error during auto-commit: ${error.message}${colors.reset}`);
    process.exit(1);
});