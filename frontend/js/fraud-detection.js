// ===================================
// FRAUD DETECTION - CLIENT LOGIC
// ===================================

// Transaction history storage
let transactionHistory = [];

// Form submission
document.getElementById('fraudForm').addEventListener('submit', function (e) {
    e.preventDefault();
    analyzeTransaction();
});

function analyzeTransaction() {
    // Get form data
    const formData = {
        amount: parseFloat(document.getElementById('amount').value),
        hour: parseInt(document.getElementById('hour').value),
        isWeekend: parseInt(document.getElementById('isWeekend').value),
        locationChange: parseInt(document.getElementById('locationChange').value),
        deviceChange: parseInt(document.getElementById('deviceChange').value),
        failedLogins: parseInt(document.getElementById('failedLogins').value)
    };

    // Calculate risk score (simulated - will be replaced with ML model)
    const riskAnalysis = calculateRiskScore(formData);

    // Display result
    displayRiskResult(riskAnalysis);

    // Add to history
    addToHistory(formData, riskAnalysis);
}

function calculateRiskScore(data) {
    let riskScore = 0;
    let factors = [];

    // Amount risk (0-30%)
    if (data.amount > 50000) {
        riskScore += 30;
        factors.push(`Large amount (₹${data.amount.toLocaleString()})`);
    } else if (data.amount > 20000) {
        riskScore += 15;
        factors.push(`Above-average amount (₹${data.amount.toLocaleString()})`);
    }

    // Time risk (0-25%)
    if (data.hour >= 0 && data.hour <= 5) {
        riskScore += 25;
        factors.push(`Unusual time (${formatHour(data.hour)})`);
    } else if (data.hour >= 22) {
        riskScore += 10;
        factors.push(`Late night (${formatHour(data.hour)})`);
    }

    // Location change (0-20%)
    if (data.locationChange === 1) {
        riskScore += 20;
        factors.push('Different location detected');
    }

    // Device change (0-20%)
    if (data.deviceChange === 1) {
        riskScore += 20;
        factors.push('New device detected');
    }

    // Failed logins (0-15%)
    if (data.failedLogins > 0) {
        const loginRisk = Math.min(data.failedLogins * 5, 15);
        riskScore += loginRisk;
        factors.push(`${data.failedLogins} failed login attempt(s)`);
    }

    // Weekend risk (0-5%)
    if (data.isWeekend === 1 && data.hour >= 0 && data.hour <= 6) {
        riskScore += 5;
        factors.push('Weekend early morning');
    }

    // Cap at 100%
    riskScore = Math.min(riskScore, 100);

    // Determine risk level
    let riskLevel, riskColor;
    if (riskScore < 30) {
        riskLevel = 'Low Risk';
        riskColor = '#10B981'; // Green
    } else if (riskScore < 70) {
        riskLevel = 'Medium Risk';
        riskColor = '#F59E0B'; // Yellow
    } else {
        riskLevel = 'High Risk';
        riskColor = '#EF4444'; // Red
    }

    return {
        score: riskScore,
        level: riskLevel,
        color: riskColor,
        factors: factors.length > 0 ? factors : ['No significant risk factors detected']
    };
}

function displayRiskResult(analysis) {
    // Show result card
    const resultCard = document.getElementById('resultCard');
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Animate risk score
    animateRiskGauge(analysis.score, analysis.color);

    // Update risk label
    document.getElementById('riskLabel').textContent = analysis.level;
    document.getElementById('riskLabel').style.color = analysis.color;

    // Display risk factors
    const factorsList = document.getElementById('riskFactors');
    factorsList.innerHTML = '';
    analysis.factors.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        factorsList.appendChild(li);
    });
}

function animateRiskGauge(targetScore, color) {
    const scoreElement = document.getElementById('riskScore');
    const riskArc = document.getElementById('riskArc');

    // Animate number
    let currentScore = 0;
    const duration = 1500; // 1.5 seconds
    const increment = targetScore / (duration / 16); // 60fps

    const animateScore = () => {
        currentScore += increment;
        if (currentScore < targetScore) {
            scoreElement.textContent = Math.floor(currentScore);
            requestAnimationFrame(animateScore);
        } else {
            scoreElement.textContent = Math.floor(targetScore);
        }
    };

    animateScore();

    // Animate arc
    const arcLength = 251.2; // Circumference of semi-circle
    const offset = arcLength - (arcLength * targetScore / 100);

    riskArc.style.stroke = color;
    riskArc.style.strokeDashoffset = offset;
    riskArc.style.transition = 'stroke-dashoffset 1.5s ease-out';
}

function addToHistory(data, analysis) {
    const transaction = {
        timestamp: new Date(),
        amount: data.amount,
        hour: data.hour,
        riskScore: analysis.score,
        riskLevel: analysis.level,
        riskColor: analysis.color
    };

    transactionHistory.unshift(transaction); // Add to beginning

    // Keep only last 10
    if (transactionHistory.length > 10) {
        transactionHistory = transactionHistory.slice(0, 10);
    }

    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    const historyContainer = document.getElementById('transactionHistory');

    if (transactionHistory.length === 0) {
        historyContainer.innerHTML = '<p class="text-muted">No transactions yet.</p>';
        return;
    }

    let html = '<table class="transaction-table"><thead><tr><th>Time</th><th>Amount</th><th>Risk Score</th><th>Status</th></tr></thead><tbody>';

    transactionHistory.forEach(tx => {
        const timeStr = tx.timestamp.toLocaleTimeString();
        const dateStr = tx.timestamp.toLocaleDateString();

        html += `
            <tr>
                <td>${dateStr} ${timeStr}</td>
                <td>₹${tx.amount.toLocaleString()}</td>
                <td><span class="risk-badge" style="background: ${tx.riskColor}20; color: ${tx.riskColor};">${tx.riskScore}%</span></td>
                <td><span class="risk-level" style="color: ${tx.riskColor};">${tx.riskLevel}</span></td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    historyContainer.innerHTML = html;
}

function resetForm() {
    document.getElementById('fraudForm').reset();
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('fraudForm').scrollIntoView({ behavior: 'smooth' });
}

function formatHour(hour) {
    if (hour === 0) return '12 AM';
    if (hour < 12) return `${hour} AM`;
    if (hour === 12) return '12 PM';
    return `${hour - 12} PM`;
}

// Initialize
console.log('🛡️ Fraud Detection System Loaded');
console.log('⚠️  Currently using simulated risk calculation');
console.log('🔄 Will be replaced with trained ML model in production');
