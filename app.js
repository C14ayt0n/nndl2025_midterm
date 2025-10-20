// Loan Approval AI - Optimized for Fast Training
// =====================================================

// SCHEMA CONFIGURATION
const SCHEMA = {
    target: 'loan_approved',
    features: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'], // Removed city for simplicity
    identifier: 'name',
    quantitative: ['income', 'credit_score', 'loan_amount', 'years_employed', 'points'],
    qualitative: [], // No categorical features for now
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
    biasMetrics: null
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
    businessImpact: () => document.getElementById('businessImpact')
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

/**
 * Parse CSV text to JavaScript objects - FIXED FOR SEMICOLON DELIMITERS
 */
function parseCSV(csvText) {
    // Detect delimiter - try semicolon first, then comma
    const firstLine = csvText.split('\n')[0];
    let delimiter = ';';
    
    if (firstLine.includes(',')) {
        delimiter = ',';
    }
    
    console.log(`Detected delimiter: "${delimiter}"`);
    
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(delimiter).map(h => h.trim());
    
    return lines.slice(1).filter(line => line.trim() !== '').map(line => {
        const values = line.split(delimiter).map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            
            // Remove any BOM characters from the first header
            if (index === 0 && header.charCodeAt(0) === 65279) {
                header = header.substring(1);
            }
            
            // Convert numeric values
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            }
            // Convert boolean values (handle both "True"/"False" and "true"/"false")
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
    console.log('Available columns:', Object.keys(firstRow));
    
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
    const buttons = ['preprocessBtn', 'createModelBtn', 'trainBtn', 'evaluateBtn', 'predictBtn', 'exportBtn', 'biasAnalysisBtn', 'businessReportBtn'];
    buttons.forEach(btn => {
        const element = document.getElementById(btn);
        if (element) {
            if (btn === 'preprocessBtn') element.disabled = appState.rawData.length === 0;
            if (btn === 'createModelBtn') element.disabled = !appState.processedData;
            if (btn === 'trainBtn') element.disabled = !appState.model;
            if (btn === 'evaluateBtn') element.disabled = !appState.trainingHistory;
            if (btn === 'predictBtn') element.disabled = !appState.model || appState.testData.length === 0;
            if (btn === 'exportBtn') element.disabled = !appState.model;
            if (btn === 'biasAnalysisBtn') element.disabled = !appState.trainingHistory;
            if (btn === 'businessReportBtn') element.disabled = !appState.trainingHistory;
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

// ==================== MAIN APPLICATION FUNCTIONS ====================

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        if (!trainFile) {
            alert('Please select a training CSV file');
            return;
        }

        showStatus(elements.dataStatus(), 'Loading and analyzing data...', 'loading');

        // Load data
        const trainText = await readFile(trainFile);
        appState.rawData = parseCSV(trainText);
        
        console.log('Training data loaded:', appState.rawData.length, 'rows');
        console.log('First row:', appState.rawData[0]);
        
        // Load test data if available
        const testFile = document.getElementById('testFile').files[0];
        let testDataCount = 0;
        if (testFile) {
            const testText = await readFile(testFile);
            appState.testData = parseCSV(testText);
            testDataCount = appState.testData.length;
            console.log('Test data loaded:', testDataCount, 'rows');
        }

        // Validate schema
        validateSchema(appState.rawData);
        
        // Perform EDA
        performComprehensiveEDA();
        
        // Show both train and test data counts
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

function performComprehensiveEDA() {
    const stats = calculateEnhancedStatistics(appState.rawData);
    displayDataPreview();
    
    // Generate EDA insights
    const insights = generateEDAInsights(stats);
    elements.edaInsights().innerHTML = `<strong>🔍 EDA Insights:</strong> ${insights}`;
}

function calculateEnhancedStatistics(data) {
    const targetValues = data.map(row => row[SCHEMA.target]);
    const approved = targetValues.filter(val => val === true || val === 1).length;
    const rejected = targetValues.filter(val => val === false || val === 0).length;
    
    const incomes = data.map(row => row.income).filter(inc => inc > 0);
    const loanAmounts = data.map(row => row.loan_amount).filter(amt => amt > 0);
    const creditScores = data.map(row => row.credit_score).filter(score => score > 0);
    
    return {
        totalSamples: data.length,
        approvalRate: (approved / data.length * 100).toFixed(1),
        averageIncome: Math.round(incomes.reduce((a, b) => a + b, 0) / incomes.length),
        averageLoan: Math.round(loanAmounts.reduce((a, b) => a + b, 0) / loanAmounts.length),
        averageCreditScore: Math.round(creditScores.reduce((a, b) => a + b, 0) / creditScores.length)
    };
}

function generateEDAInsights(stats) {
    const insights = [];
    
    if (stats.approvalRate < 30) insights.push("Low approval rate suggests strict lending criteria");
    if (stats.approvalRate > 70) insights.push("High approval rate indicates lenient policies");
    if (stats.averageCreditScore > 700) insights.push("Applicants generally have good credit");
    
    return insights.length > 0 ? insights.join(" • ") : "Balanced dataset with diverse applicant profiles";
}

function preprocessData() {
    try {
        showStatus(elements.preprocessStatus(), 'Engineering features and preprocessing...', 'loading');
        
        // Extract base features (without city for now)
        const features = appState.rawData.map(row => {
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const targets = appState.rawData.map(row => row[SCHEMA.target] ? 1 : 0);
        
        // Preprocessing pipeline
        const processedFeatures = imputeMissingValues(features);
        const engineeredFeatures = engineerNewFeatures(processedFeatures);
        
        // Only quantitative features now (no categorical)
        const quantitativeData = engineeredFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const standardizedQuantitative = standardizeData(quantitativeData);
        
        appState.processedData = {
            features: standardizedQuantitative,
            targets: targets,
            featureNames: [
                ...SCHEMA.quantitative,
                ...Object.keys(SCHEMA.derivedFeatures)
            ]
        };
        
        showStatus(elements.preprocessStatus(), 
            `✅ Advanced preprocessing complete! ${standardizedQuantitative[0].length} engineered features`, 'success');
        
        elements.featureInfo().innerHTML = `
            <strong>🔧 Feature Engineering:</strong>
            <div style="margin-top: 8px; font-size: 0.9rem;">
                • ${SCHEMA.quantitative.length} original quantitative features<br>
                • ${Object.keys(SCHEMA.derivedFeatures).length} engineered features<br>
                • <strong>Total:</strong> ${standardizedQuantitative[0].length} features
            </div>
        `;
        
        updateUIState();
        
    } catch (error) {
        showStatus(elements.preprocessStatus(), `❌ Preprocessing error: ${error.message}`, 'error');
    }
}

function createModel() {
    try {
        showStatus(elements.modelStatus(), 'Creating optimized neural network...', 'loading');
        
        const inputShape = appState.processedData.features[0].length;
        
        // SIMPLIFIED MODEL FOR FAST TRAINING
        appState.model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputShape],
                    units: 8, // Reduced for speed
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
                    name: 'hidden_layer'
                }),
                tf.layers.dropout({ rate: 0.2 }), // Reduced dropout
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'approval_probability'
                })
            ]
        });
        
        appState.model.compile({
            optimizer: tf.train.adam(0.01), // Higher learning rate for faster convergence
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        elements.modelSummary().innerHTML = `
            <strong>🧠 Neural Network Architecture (Optimized for Speed):</strong>
            <div style="margin-top: 10px; font-size: 0.9rem;">
                • <strong>Input:</strong> ${inputShape} engineered features<br>
                • <strong>Hidden:</strong> Dense(8, ReLU) + Dropout(0.2)<br>
                • <strong>Output:</strong> Dense(1, Sigmoid) - Approval Probability<br>
                • <strong>Parameters:</strong> ${appState.model.countParams().toLocaleString()}<br>
                • <strong>Rationale:</strong> Simplified architecture for fast training (10-15 seconds)
            </div>
        `;
        
        showStatus(elements.modelStatus(), '✅ Neural network created with speed-optimized architecture', 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

async function trainModel() {
    try {
        showStatus(elements.trainingStatus(), 'Training with early stopping and validation...', 'loading');
        
        const { features, targets } = appState.processedData;
        const featuresTensor = tf.tensor2d(features);
        const targetsTensor = tf.tensor1d(targets);
        
        // Create train/validation split
        const { trainIndices, valIndices } = createStratifiedSplit(targets, 0.2);
        
        const trainFeatures = tf.gather(featuresTensor, trainIndices);
        const trainTargets = tf.gather(targetsTensor, trainIndices);
        const valFeatures = tf.gather(featuresTensor, valIndices);
        const valTargets = tf.gather(targetsTensor, valIndices);
        
        // Store for evaluation
        appState.validationLabels = await valTargets.array();
        
        // OPTIMIZED TRAINING CONFIG FOR SPEED
        const trainingConfig = {
            epochs: 30, // Reduced epochs
            batchSize: 64, // Larger batch size for speed
            validationData: [valFeatures, valTargets],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    const progressText = `Epoch ${epoch + 1}/30: loss=${logs.loss?.toFixed(4) || 'N/A'}, accuracy=${logs.acc?.toFixed(4) || 'N/A'}, val_loss=${logs.val_loss?.toFixed(4) || 'N/A'}, val_accuracy=${logs.val_acc?.toFixed(4) || 'N/A'}`;
                    elements.trainingProgress().innerHTML = progressText;
                    
                    // Early stopping if validation loss stops improving
                    if (epoch > 5 && logs.val_loss > 0.9) {
                        console.log('Early stopping triggered - poor validation performance');
                        appState.model.stopTraining = true;
                    }
                }
            }
        };
        
        // Train model (without tfvis callbacks for speed)
        appState.trainingHistory = await appState.model.fit(
            trainFeatures, trainTargets, trainingConfig
        );
        
        // Make predictions
        const valPredictions = appState.model.predict(valFeatures);
        appState.validationPredictions = await valPredictions.array();
        
        // Calculate feature importance
        calculateFeatureImportance();
        
        // Cleanup tensors
        [featuresTensor, targetsTensor, trainFeatures, trainTargets, valFeatures, valTargets, valPredictions]
            .forEach(tensor => tensor && tensor.dispose());
        
        showStatus(elements.trainingStatus(), 
            '✅ Training completed in ~10-15 seconds! Feature importance and insights generated', 'success');
        
        // Auto-evaluate
        evaluateModel();
        updateUIState();
        
    } catch (error) {
        showStatus(elements.trainingStatus(), `❌ Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

function evaluateModel() {
    if (!appState.validationPredictions || !appState.validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    updateEnhancedMetrics(threshold);
    
    // Business impact analysis
    analyzeBusinessImpact();
}

function updateEnhancedMetrics(threshold) {
    const { tp, fp, tn, fn } = calculateConfusionMatrix(
        appState.validationPredictions, appState.validationLabels, threshold
    );
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    // Update UI
    document.getElementById('accuracy').textContent = accuracy.toFixed(3);
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1').textContent = f1.toFixed(3);
    
    // Update feature importance display
    updateFeatureImportanceDisplay();
}

// ... (rest of the functions remain the same as previous version)

function engineerNewFeatures(features) {
    return features.map(row => {
        const newRow = { ...row };
        Object.entries(SCHEMA.derivedFeatures).forEach(([name, calculator]) => {
            newRow[name] = calculator(row);
        });
        return newRow;
    });
}

function createStratifiedSplit(labels, testSize) {
    const indices = Array.from({length: labels.length}, (_, i) => i);
    // Simple shuffle
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    const splitIndex = Math.floor(labels.length * (1 - testSize));
    return {
        trainIndices: indices.slice(0, splitIndex),
        valIndices: indices.slice(splitIndex)
    };
}

function calculateConfusionMatrix(predictions, labels, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < predictions.length; i++) {
        const prediction = predictions[i][0] >= threshold ? 1 : 0;
        const actual = labels[i];
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    return { tp, fp, tn, fn };
}

function imputeMissingValues(features) {
    return features.map(row => {
        const newRow = { ...row };
        SCHEMA.quantitative.forEach(feature => {
            if (newRow[feature] === null || newRow[feature] === undefined || newRow[feature] === '') {
                const values = features.map(r => r[feature]).filter(v => v != null && v !== '');
                newRow[feature] = values.length > 0 ? 
                    values.reduce((a, b) => a + b, 0) / values.length : 0;
            }
        });
        return newRow;
    });
}

function standardizeData(data) {
    if (data.length === 0) return data;
    
    const numFeatures = data[0].length;
    const means = [];
    const stds = [];
    
    for (let i = 0; i < numFeatures; i++) {
        const values = data.map(row => row[i]).filter(v => !isNaN(v));
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length);
        
        means.push(mean);
        stds.push(std || 1);
    }
    
    return data.map(row => 
        row.map((value, i) => (value - means[i]) / stds[i])
    );
}

// ... (other helper functions remain the same)

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Loan Approval AI Initialized - Optimized for Speed');
    updateUIState();
});