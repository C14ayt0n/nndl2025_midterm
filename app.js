// Loan Approval AI - Optimized for Speed & Functionality
// =====================================================

// SCHEMA CONFIGURATION
const SCHEMA = {
    target: 'loan_approved',
    features: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    identifier: 'name',
    quantitative: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    qualitative: [],
    derivedFeatures: {
        'debt_to_income': (row) => row.loan_amount / (row.income || 1),
        'credit_utilization': (row) => row.points / 100,
        'loan_to_income': (row) => row.loan_amount / (row.income || 1)
    }
};

// Global application state
const appState = {
    rawData: [],
    testData: [],
    processedData: null,
    model: null,
    trainingHistory: null,
    validationPredictions: null,
    validationLabels: null,
    featureImportance: null,
    charts: {},
    currentChart: null,
    testPredictions: null,
    preprocessingStats: null
};

// DOM elements
const elements = {
    dataStatus: () => document.getElementById('dataStatus'),
    preprocessStatus: () => document.getElementById('preprocessStatus'),
    modelStatus: () => document.getElementById('modelStatus'),
    trainingStatus: () => document.getElementById('trainingStatus'),
    predictionStatus: () => document.getElementById('predictionStatus'),
    dataPreview: () => document.getElementById('dataPreview'),
    featureInfo: () => document.getElementById('featureInfo'),
    modelSummary: () => document.getElementById('modelSummary'),
    trainingProgress: () => document.getElementById('trainingProgress'),
    thresholdValue: () => document.getElementById('thresholdValue'),
    featureImportance: () => document.getElementById('featureImportance'),
    edaInsights: () => document.getElementById('edaInsights'),
    chartsContainer: () => document.getElementById('chartsContainer'),
    chartControls: () => document.getElementById('chartControls'),
    chartDisplay: () => document.getElementById('chartDisplay'),
    predictionResults: () => document.getElementById('predictionResults'),
    preprocessingDetails: () => document.getElementById('preprocessingDetails')
};

// ==================== UTILITY FUNCTIONS ====================

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    const firstLine = csvText.split('\n')[0];
    let delimiter = ';';
    
    if (firstLine.includes(',')) {
        delimiter = ',';
    }
    
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(delimiter).map(h => h.trim());
    
    return lines.slice(1).filter(line => line.trim() !== '').map(line => {
        const values = line.split(delimiter).map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            
            if (index === 0 && header.charCodeAt(0) === 65279) {
                header = header.substring(1);
            }
            
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            }
            else if (value === 'true' || value === '1' || value === 'True') {
                value = true;
            }
            else if (value === 'false' || value === '0' || value === 'False') {
                value = false;
            }
            
            row[header] = value;
        });
        return row;
    });
}

function validateSchema(data) {
    if (data.length === 0) {
        throw new Error('No data loaded');
    }
    
    const firstRow = data[0];
    
    SCHEMA.features.forEach(feature => {
        if (!(feature in firstRow)) {
            throw new Error(`Missing required feature: ${feature}. Available: ${Object.keys(firstRow).join(', ')}`);
        }
    });
    
    if (!(SCHEMA.target in firstRow)) {
        throw new Error(`Missing target variable: ${SCHEMA.target}. Available: ${Object.keys(firstRow).join(', ')}`);
    }
}

function showStatus(element, message, type = '') {
    element.textContent = message;
    element.className = `status ${type}`;
}

function updateUIState() {
    const buttons = ['preprocessBtn', 'createModelBtn', 'trainBtn', 'predictBtn', 'exportBtn', 'downloadPredictionsBtn'];
    buttons.forEach(btn => {
        const element = document.getElementById(btn);
        if (element) {
            if (btn === 'preprocessBtn') element.disabled = appState.rawData.length === 0;
            if (btn === 'createModelBtn') element.disabled = !appState.processedData;
            if (btn === 'trainBtn') element.disabled = !appState.model;
            if (btn === 'predictBtn') element.disabled = !appState.model || appState.testData.length === 0;
            if (btn === 'exportBtn') element.disabled = !appState.model;
            if (btn === 'downloadPredictionsBtn') element.disabled = !appState.testPredictions;
        }
    });
}

function displayDataPreview() {
    const container = elements.dataPreview();
    const data = appState.rawData.slice(0, 10);
    
    if (data.length === 0) {
        container.innerHTML = '<p>No data to display</p>';
        return;
    }
    
    const headers = Object.keys(data[0]);
    let html = `<table>
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>`;
    
    data.forEach(row => {
        html += `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ==================== EDA & VISUALIZATION FUNCTIONS ====================

function performComprehensiveEDA() {
    const stats = calculateEnhancedStatistics(appState.rawData);
    displayDataPreview();
    createInteractiveCharts(appState.rawData);
    
    const insights = generateDetailedEDAInsights(stats);
    elements.edaInsights().innerHTML = insights;
}

function calculateEnhancedStatistics(data) {
    const targetValues = data.map(row => row[SCHEMA.target]);
    const approved = targetValues.filter(val => val === true || val === 1).length;
    const rejected = targetValues.filter(val => val === false || val === 0).length;
    
    const missingValues = {};
    SCHEMA.features.forEach(feature => {
        missingValues[feature] = data.filter(row => 
            row[feature] === null || row[feature] === undefined || row[feature] === ''
        ).length;
    });
    
    const numericalStats = {};
    SCHEMA.quantitative.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val != null && val !== '');
        if (values.length > 0) {
            numericalStats[feature] = {
                min: Math.min(...values),
                max: Math.max(...values),
                mean: values.reduce((a, b) => a + b, 0) / values.length,
                median: calculateMedian(values),
                missing: missingValues[feature],
                missingPercentage: ((missingValues[feature] / data.length) * 100).toFixed(1)
            };
        }
    });
    
    return {
        totalSamples: data.length,
        approvalRate: (approved / data.length * 100).toFixed(1),
        approvedCount: approved,
        rejectedCount: rejected,
        numericalStats: numericalStats,
        missingValues: missingValues
    };
}

function calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function createInteractiveCharts(data) {
    // Clear previous charts
    elements.chartControls().innerHTML = '';
    elements.chartDisplay().innerHTML = '';
    
    // Properly destroy existing charts
    Object.values(appState.charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    appState.charts = {};
    
    // Create chart buttons
    const chartTypes = ['Target Distribution', ...SCHEMA.quantitative];
    
    chartTypes.forEach((type, index) => {
        const button = document.createElement('button');
        button.className = `chart-btn ${index === 0 ? 'active' : ''}`;
        button.textContent = type;
        button.onclick = () => showChart(type);
        elements.chartControls().appendChild(button);
    });
    
    // Create all charts
    createTargetDistributionChart(data);
    SCHEMA.quantitative.forEach(feature => {
        createFeatureDistributionChart(data, feature);
    });
    
    // Show first chart
    showChart('Target Distribution');
}

function showChart(chartType) {
    // Update button states
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent === chartType) {
            btn.classList.add('active');
        }
    });
    
    // Clear display and create new canvas
    elements.chartDisplay().innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = 500;
    canvas.height = 300;
    elements.chartDisplay().appendChild(canvas);
    
    // Create the chart on the canvas
    if (appState.charts[chartType] && appState.charts[chartType].config) {
        const ctx = canvas.getContext('2d');
        appState.charts[chartType].instance = new Chart(ctx, appState.charts[chartType].config);
    }
}

function createTargetDistributionChart(data) {
    const targetValues = data.map(row => row[SCHEMA.target]);
    const approved = targetValues.filter(val => val === true || val === 1).length;
    const rejected = targetValues.filter(val => val === false || val === 0).length;
    
    const config = {
        type: 'doughnut',
        data: {
            labels: ['Approved', 'Rejected'],
            datasets: [{
                data: [approved, rejected],
                backgroundColor: ['#27ae60', '#e74c3c'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Loan Approval Distribution'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    };
    
    appState.charts['Target Distribution'] = { config: config };
}

function createFeatureDistributionChart(data, feature) {
    const values = data.map(row => row[feature]).filter(v => v != null);
    
    // Create bins for histogram
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binCount = Math.min(15, Math.floor(values.length / 20));
    const binSize = (max - min) / binCount;
    
    const bins = Array(binCount).fill(0);
    values.forEach(value => {
        const binIndex = Math.min(binCount - 1, Math.floor((value - min) / binSize));
        bins[binIndex]++;
    });
    
    const config = {
        type: 'bar',
        data: {
            labels: bins.map((_, i) => {
                const start = min + i * binSize;
                const end = min + (i + 1) * binSize;
                return `${Math.round(start)}-${Math.round(end)}`;
            }),
            datasets: [{
                label: `Count`,
                data: bins,
                backgroundColor: '#3498db',
                borderColor: '#2980b9',
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            plugins: {
                title: {
                    display: true,
                    text: `${feature} Distribution`
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: feature
                    }
                }
            }
        }
    };
    
    appState.charts[feature] = { config: config };
}

function generateDetailedEDAInsights(stats) {
    let insights = `<strong>🔍 Detailed EDA Analysis:</strong><div style="margin-top: 10px;">`;
    
    insights += `<div style="margin-bottom: 10px;">
        <strong>Dataset Overview:</strong><br>
        • Total samples: ${stats.totalSamples}<br>
        • Approval rate: ${stats.approvalRate}% (${stats.approvedCount} approved, ${stats.rejectedCount} rejected)
    </div>`;
    
    let hasMissing = false;
    insights += `<div style="margin-bottom: 10px;">
        <strong>Missing Values Analysis:</strong><br>`;
    
    SCHEMA.features.forEach(feature => {
        if (stats.missingValues[feature] > 0) {
            insights += `• ${feature}: ${stats.missingValues[feature]} (${stats.numericalStats[feature]?.missingPercentage || '0'}%)<br>`;
            hasMissing = true;
        }
    });
    
    if (!hasMissing) {
        insights += `• No missing values detected<br>`;
    }
    insights += `</div>`;
    
    insights += `<div>
        <strong>Business Insights:</strong><br>`;
    
    if (stats.approvalRate < 30) insights += "• Low approval rate suggests strict lending criteria<br>";
    if (stats.approvalRate > 70) insights += "• High approval rate indicates lenient policies<br>";
    if (stats.numericalStats.credit_score?.mean > 700) insights += "• Applicants generally have good credit scores<br>";
    
    if (!insights.includes("Business Insights")) {
        insights += "• Balanced dataset with diverse applicant profiles";
    }
    
    insights += `</div></div>`;
    return insights;
}

// ==================== MAIN APPLICATION FUNCTIONS ====================

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        if (!trainFile) {
            alert('Please select a training CSV file');
            return;
        }

        showStatus(elements.dataStatus(), 'Loading and analyzing data...', 'loading');

        const trainText = await readFile(trainFile);
        appState.rawData = parseCSV(trainText);
        
        const testFile = document.getElementById('testFile').files[0];
        let testDataCount = 0;
        if (testFile) {
            const testText = await readFile(testFile);
            appState.testData = parseCSV(testText);
            testDataCount = appState.testData.length;
        }

        validateSchema(appState.rawData);
        performComprehensiveEDA();
        
        const statusMessage = testDataCount > 0 
            ? `✅ Data loaded! ${appState.rawData.length} training samples, ${testDataCount} test samples, ${Object.keys(appState.rawData[0]).length} features`
            : `✅ Data loaded! ${appState.rawData.length} samples, ${Object.keys(appState.rawData[0]).length} features`;
            
        showStatus(elements.dataStatus(), statusMessage, 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.dataStatus(), `❌ Data loading error: ${error.message}`, 'error');
        console.error('Data loading error:', error);
    }
}

function preprocessData() {
    try {
        showStatus(elements.preprocessStatus(), 'Engineering features and preprocessing...', 'loading');
        
        const features = appState.rawData.map(row => {
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const targets = appState.rawData.map(row => row[SCHEMA.target] ? 1 : 0);
        
        // Track preprocessing statistics
        appState.preprocessingStats = {
            originalFeatures: SCHEMA.features.length,
            missingValues: {},
            standardization: {}
        };
        
        const processedFeatures = imputeMissingValues(features);
        const engineeredFeatures = engineerNewFeatures(processedFeatures);
        
        const quantitativeData = engineeredFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const { standardizedData, means, stds } = standardizeDataWithStats(quantitativeData);
        
        // Store standardization stats
        SCHEMA.quantitative.forEach((feature, index) => {
            appState.preprocessingStats.standardization[feature] = {
                mean: means[index],
                std: stds[index]
            };
        });
        
        appState.processedData = {
            features: standardizedData,
            targets: targets,
            featureNames: [
                ...SCHEMA.quantitative,
                ...Object.keys(SCHEMA.derivedFeatures)
            ]
        };
        
        showPreprocessingDetails();
        showStatus(elements.preprocessStatus(), 
            `✅ Advanced preprocessing complete! ${standardizedData[0].length} engineered features`, 'success');
        
        elements.featureInfo().innerHTML = `
            <strong>🔧 Feature Engineering:</strong>
            <div style="margin-top: 8px;">
                • Original features: ${SCHEMA.features.length}<br>
                • Engineered features: ${Object.keys(SCHEMA.derivedFeatures).length}<br>
                • Total features: ${standardizedData[0].length}<br>
                • Samples: ${standardizedData.length}
            </div>
        `;
        
        updateUIState();
        
    } catch (error) {
        showStatus(elements.preprocessStatus(), `❌ Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

function imputeMissingValues(features) {
    const processed = JSON.parse(JSON.stringify(features));
    
    SCHEMA.quantitative.forEach(feature => {
        const values = processed.map(row => row[feature]).filter(val => val != null);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        
        processed.forEach(row => {
            if (row[feature] == null) {
                row[feature] = mean;
            }
        });
    });
    
    return processed;
}

function engineerNewFeatures(features) {
    return features.map(row => {
        const newRow = { ...row };
        Object.entries(SCHEMA.derivedFeatures).forEach(([name, fn]) => {
            newRow[name] = fn(row);
        });
        return newRow;
    });
}

function standardizeDataWithStats(data) {
    const means = [];
    const stds = [];
    
    const transposed = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
    
    const standardizedTransposed = transposed.map((column, index) => {
        const validValues = column.filter(val => !isNaN(val));
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const std = Math.sqrt(validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length);
        
        means[index] = mean;
        stds[index] = std || 1;
        
        return column.map(val => isNaN(val) ? 0 : (val - mean) / std);
    });
    
    const standardizedData = standardizedTransposed[0].map((_, rowIndex) => 
        standardizedTransposed.map(column => column[rowIndex])
    );
    
    return { standardizedData, means, stds };
}

function showPreprocessingDetails() {
    if (!appState.preprocessingStats) return;
    
    let details = `<strong>📊 Preprocessing Details:</strong><br>`;
    details += `• Features engineered: ${Object.keys(SCHEMA.derivedFeatures).length}<br>`;
    details += `• Standardization applied to ${SCHEMA.quantitative.length} features<br>`;
    
    if (Object.keys(appState.preprocessingStats.standardization).length > 0) {
        details += `<br><strong>Standardization Parameters:</strong><br>`;
        Object.entries(appState.preprocessingStats.standardization).slice(0, 3).forEach(([feature, stats]) => {
            details += `• ${feature}: mean=${stats.mean.toFixed(2)}, std=${stats.std.toFixed(2)}<br>`;
        });
        if (Object.keys(appState.preprocessingStats.standardization).length > 3) {
            details += `• ... and ${Object.keys(appState.preprocessingStats.standardization).length - 3} more features`;
        }
    }
    
    elements.preprocessingDetails().innerHTML = details;
}

function createModel() {
    try {
        showStatus(elements.modelStatus(), 'Creating neural network architecture...', 'loading');
        
        appState.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [appState.processedData.featureNames.length],
                    units: 12,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        appState.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy', 'precision', 'recall']
        });
        
        const summary = [];
        appState.model.summary(1, (_, strings) => {
            summary.push(...strings);
        });
        
        elements.modelSummary().innerHTML = `
            <strong>🧠 Neural Network Architecture:</strong>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.8rem; margin-top: 10px; overflow-x: auto;">
${summary.join('\n')}
            </pre>
            <div style="margin-top: 10px;">
                <strong>Key Features:</strong><br>
                • Input: ${appState.processedData.featureNames.length} features<br>
                • Hidden: 12 units with ReLU + Dropout (30%)<br>
                • Output: 1 unit with Sigmoid activation<br>
                • Optimizer: Adam (learning rate: 0.001)
            </div>
        `;
        
        showStatus(elements.modelStatus(), '✅ Neural network created successfully!', 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

async function trainModel() {
    try {
        showStatus(elements.trainingStatus(), 'Training neural network...', 'loading');
        
        const features = tf.tensor2d(appState.processedData.features);
        const targets = tf.tensor1d(appState.processedData.targets);
        
        const validationSplit = 0.2;
        const batchSize = 32;
        const epochs = 20;
        
        elements.trainingProgress().innerHTML = '<div style="margin: 10px 0;">Training progress will appear here...</div>';
        
        const history = await appState.model.fit(features, targets, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: validationSplit,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = `
                        <div style="background: white; padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #3498db;">
                            Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${logs.acc.toFixed(4)} - Val Loss: ${logs.val_loss.toFixed(4)} - Val Acc: ${logs.val_acc.toFixed(4)}
                        </div>
                    `;
                    elements.trainingProgress().innerHTML += progress;
                }
            }
        });
        
        appState.trainingHistory = history;
        
        const validationData = getValidationData(features, targets, validationSplit);
        const predictions = appState.model.predict(validationData.features);
        const predictedValues = await predictions.data();
        
        appState.validationPredictions = predictedValues;
        appState.validationLabels = await validationData.labels.data();
        
        calculateAndDisplayMetrics();
        calculateFeatureImportance();
        
        showStatus(elements.trainingStatus(), 
            `✅ Training complete! Final validation accuracy: ${(history.history.val_acc[history.history.val_acc.length - 1] * 100).toFixed(1)}%`, 
            'success');
        
        updateUIState();
        
        tf.dispose([features, targets, validationData.features, validationData.labels, predictions]);
        
    } catch (error) {
        showStatus(elements.trainingStatus(), `❌ Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

function getValidationData(features, targets, validationSplit) {
    const numSamples = features.shape[0];
    const numValidation = Math.floor(numSamples * validationSplit);
    
    const validationFeatures = features.slice(numSamples - numValidation);
    const validationLabels = targets.slice(numSamples - numValidation);
    
    return { features: validationFeatures, labels: validationLabels };
}

function calculateAndDisplayMetrics() {
    if (!appState.validationPredictions || !appState.validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    const binaryPredictions = appState.validationPredictions.map(p => p >= threshold ? 1 : 0);
    const actualLabels = appState.validationLabels;
    
    let truePositives = 0, falsePositives = 0, trueNegatives = 0, falseNegatives = 0;
    
    for (let i = 0; i < binaryPredictions.length; i++) {
        const predicted = binaryPredictions[i];
        const actual = actualLabels[i];
        
        if (predicted === 1 && actual === 1) truePositives++;
        else if (predicted === 1 && actual === 0) falsePositives++;
        else if (predicted === 0 && actual === 0) trueNegatives++;
        else if (predicted === 0 && actual === 1) falseNegatives++;
    }
    
    const accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    document.getElementById('accuracy').textContent = accuracy.toFixed(3);
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1').textContent = f1.toFixed(3);
    
    const metricsStatus = `
        <strong>📊 Real-time Metrics (Threshold: ${threshold}):</strong><br>
        • True Positives: ${truePositives}<br>
        • False Positives: ${falsePositives}<br>
        • True Negatives: ${trueNegatives}<br>
        • False Negatives: ${falseNegatives}<br>
        • Total Samples: ${binaryPredictions.length}
    `;
    
    elements.trainingStatus().innerHTML += `<div style="margin-top: 10px;">${metricsStatus}</div>`;
}

function calculateFeatureImportance() {
    if (!appState.model || !appState.processedData) return;
    
    const baselineFeatures = tf.tensor2d(appState.processedData.features);
    const baselinePredictions = appState.model.predict(baselineFeatures);
    const baselineOutput = baselinePredictions.dataSync();
    
    const importanceScores = [];
    const numFeatures = appState.processedData.featureNames.length;
    
    for (let i = 0; i < numFeatures; i++) {
        const perturbedFeatures = baselineFeatures.arraySync();
        perturbedFeatures.forEach(row => {
            const original = row[i];
            row[i] = 0;
        });
        
        const perturbedTensor = tf.tensor2d(perturbedFeatures);
        const perturbedPredictions = appState.model.predict(perturbedTensor);
        const perturbedOutput = perturbedPredictions.dataSync();
        
        let diff = 0;
        for (let j = 0; j < baselineOutput.length; j++) {
            diff += Math.abs(baselineOutput[j] - perturbedOutput[j]);
        }
        
        importanceScores.push({
            feature: appState.processedData.featureNames[i],
            importance: diff / baselineOutput.length
        });
        
        tf.dispose([perturbedTensor, perturbedPredictions]);
    }
    
    tf.dispose([baselineFeatures, baselinePredictions]);
    
    importanceScores.sort((a, b) => b.importance - a.importance);
    appState.featureImportance = importanceScores;
    
    displayFeatureImportance();
}

function displayFeatureImportance() {
    if (!appState.featureImportance) return;
    
    const maxImportance = Math.max(...appState.featureImportance.map(item => item.importance));
    
    let html = `<strong>🎯 Feature Importance Analysis:</strong><br><br>`;
    
    appState.featureImportance.slice(0, 8).forEach(item => {
        const percentage = (item.importance / maxImportance * 100).toFixed(1);
        html += `
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>${item.feature}</span>
                    <span style="font-weight: bold;">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    
    elements.featureImportance().innerHTML = html;
}

function updateThreshold(value) {
    elements.thresholdValue().textContent = value;
    if (appState.validationPredictions && appState.validationLabels) {
        calculateAndDisplayMetrics();
    }
}

async function predictTestData() {
    try {
        if (appState.testData.length === 0) {
            alert('No test data loaded. Please load test data first.');
            return;
        }
        
        showStatus(elements.predictionStatus(), 'Processing test data and generating predictions...', 'loading');
        
        const testFeatures = appState.testData.map(row => {
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const processedTestFeatures = imputeMissingValues(testFeatures);
        const engineeredTestFeatures = engineerNewFeatures(processedTestFeatures);
        
        const quantitativeTestData = engineeredTestFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const standardizedTestData = standardizeTestData(quantitativeTestData);
        
        const testTensor = tf.tensor2d(standardizedTestData);
        const predictions = appState.model.predict(testTensor);
        const predictionValues = await predictions.data();
        
        appState.testPredictions = predictionValues.map((prob, index) => ({
            identifier: appState.testData[index][SCHEMA.identifier] || `Sample_${index + 1}`,
            probability: prob,
            prediction: prob >= parseFloat(document.getElementById('thresholdSlider').value) ? 'APPROVED' : 'REJECTED',
            confidence: (prob >= 0.5 ? prob : 1 - prob) * 100
        }));
        
        displayPredictionResults();
        
        showStatus(elements.predictionStatus(), 
            `✅ Predictions generated! ${appState.testPredictions.length} samples processed`, 'success');
        
        updateUIState();
        
        tf.dispose([testTensor, predictions]);
        
    } catch (error) {
        showStatus(elements.predictionStatus(), `❌ Prediction error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

function standardizeTestData(testData) {
    if (!appState.preprocessingStats) return testData;
    
    return testData.map(row => {
        return row.map((value, index) => {
            const featureName = [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)][index];
            const stats = appState.preprocessingStats.standardization[featureName];
            if (stats) {
                return (value - stats.mean) / stats.std;
            }
            return value;
        });
    });
}

function displayPredictionResults() {
    if (!appState.testPredictions) return;
    
    const results = appState.testPredictions.slice(0, 15);
    const approvedCount = appState.testPredictions.filter(p => p.prediction === 'APPROVED').length;
    const approvalRate = (approvedCount / appState.testPredictions.length * 100).toFixed(1);
    
    let html = `
        <div style="background: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>📈 Prediction Summary:</strong><br>
            • Total predictions: ${appState.testPredictions.length}<br>
            • Approved: ${approvedCount} (${approvalRate}%)<br>
            • Rejected: ${appState.testPredictions.length - approvedCount}
        </div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Probability</th>
                    <th>Decision</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    results.forEach(pred => {
        const confidenceColor = pred.confidence > 80 ? '#27ae60' : pred.confidence > 60 ? '#f39c12' : '#e74c3c';
        html += `
            <tr>
                <td>${pred.identifier}</td>
                <td>${pred.probability.toFixed(4)}</td>
                <td style="color: ${pred.prediction === 'APPROVED' ? '#27ae60' : '#e74c3c'}; font-weight: bold;">
                    ${pred.prediction}
                </td>
                <td style="color: ${confidenceColor};">${pred.confidence.toFixed(1)}%</td>
            </tr>
        `;
    });
    
    html += `</tbody></table>`;
    
    if (appState.testPredictions.length > 15) {
        html += `<div style="margin-top: 10px; text-align: center; color: #666;">
            ... and ${appState.testPredictions.length - 15} more predictions
        </div>`;
    }
    
    elements.predictionResults().innerHTML = html;
}

function downloadPredictions() {
    if (!appState.testPredictions) return;
    
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    
    let csvContent = "ID,Approval_Probability,Predicted_Class,Confidence,Decision_Threshold\n";
    
    appState.testPredictions.forEach(pred => {
        csvContent += `${pred.identifier},${pred.probability.toFixed(6)},${pred.prediction},${pred.confidence.toFixed(2)},${threshold}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `loan_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function exportModelWeights() {
    if (!appState.model) {
        alert('No model to export');
        return;
    }
    
    const weights = appState.model.getWeights();
    const weightInfo = weights.map((weight, index) => {
        const shape = weight.shape;
        const size = shape.reduce((a, b) => a * b, 1);
        return `Layer ${index}: Shape [${shape}] - ${size} parameters`;
    }).join('\n');
    
    const blob = new Blob([weightInfo], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_weights_info.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    alert('Model weights information exported. In a real application, you would save the actual weights.');
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Loan Approval AI Application Initialized');
    console.log('📊 Features:', SCHEMA.features);
    console.log('🎯 Target:', SCHEMA.target);
    
    // Set up initial UI state
    updateUIState();
});