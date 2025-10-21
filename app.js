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
        'debt_to_income': (row) => row.loan_amount / (row.income || 1)
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
    preprocessingStats: null,
    currentMetrics: null,
    standardizationParams: null
};

// Make functions global by attaching to window object
window.appState = appState;
window.SCHEMA = SCHEMA;

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

window.readFile = function(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
};

window.parseCSV = function(csvText) {
    console.log('Parsing CSV...');
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length < 2) {
        throw new Error('CSV file must contain at least a header and one data row');
    }

    // Detect delimiter
    const firstLine = lines[0];
    let delimiter = firstLine.includes(';') ? ';' : ',';
    console.log('Using delimiter:', delimiter);

    const headers = firstLine.split(delimiter).map(h => h.trim().replace(/^\uFEFF/, '')); // Remove BOM
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim() === '') continue;
        
        const values = line.split(delimiter).map(v => v.trim());
        const row = {};
        
        headers.forEach((header, index) => {
            let value = values[index] || '';
            
            // Convert to number if possible
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            }
            // Convert boolean strings
            else if (value.toLowerCase() === 'true' || value === '1') {
                value = true;
            }
            else if (value.toLowerCase() === 'false' || value === '0') {
                value = false;
            }
            
            row[header] = value;
        });
        
        data.push(row);
    }
    
    console.log('Parsed', data.length, 'rows');
    return data;
};

window.validateSchema = function(data) {
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
};

window.showStatus = function(element, message, type = '') {
    if (element && element.textContent !== undefined) {
        element.textContent = message;
        element.className = `status ${type}`;
    }
};

window.updateUIState = function() {
    const buttons = ['preprocessBtn', 'createModelBtn', 'trainBtn', 'predictBtn', 'exportBtn', 'downloadPredictionsBtn'];
    buttons.forEach(btnId => {
        const element = document.getElementById(btnId);
        if (element) {
            if (btnId === 'preprocessBtn') element.disabled = appState.rawData.length === 0;
            if (btnId === 'createModelBtn') element.disabled = !appState.processedData;
            if (btnId === 'trainBtn') element.disabled = !appState.model;
            if (btnId === 'predictBtn') element.disabled = !appState.model || appState.testData.length === 0;
            if (btnId === 'exportBtn') element.disabled = !appState.model;
            if (btnId === 'downloadPredictionsBtn') element.disabled = !appState.testPredictions;
        }
    });
};

window.displayDataPreview = function() {
    const container = elements.dataPreview();
    if (!container) return;
    
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
};

// ==================== DATA PROCESSING FUNCTIONS ====================

window.shuffleArray = function(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
};

window.trainTestSplit = function(features, targets, testSize = 0.2) {
    const shuffledIndices = shuffleArray(Array.from({length: features.length}, (_, i) => i));
    const testCount = Math.floor(features.length * testSize);
    
    const trainIndices = shuffledIndices.slice(testCount);
    const testIndices = shuffledIndices.slice(0, testCount);
    
    const trainFeatures = trainIndices.map(i => features[i]);
    const trainTargets = trainIndices.map(i => targets[i]);
    const testFeatures = testIndices.map(i => features[i]);
    const testTargets = testIndices.map(i => targets[i]);
    
    return {
        trainFeatures,
        trainTargets,
        testFeatures,
        testTargets,
        trainIndices,
        testIndices
    };
};

// ==================== EDA FUNCTIONS ====================

window.performComprehensiveEDA = function() {
    const stats = calculateEnhancedStatistics(appState.rawData);
    displayDataPreview();
    createInteractiveCharts(appState.rawData);
    
    const insights = generateDetailedEDAInsights(stats);
    if (elements.edaInsights()) {
        elements.edaInsights().innerHTML = insights;
    }
};

window.calculateEnhancedStatistics = function(data) {
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
};

window.calculateMedian = function(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// ==================== MAIN APPLICATION FUNCTIONS ====================

window.loadData = async function() {
    try {
        console.log('Load Data function called');
        
        const trainFile = document.getElementById('trainFile').files[0];
        if (!trainFile) {
            alert('Please select a training CSV file');
            return;
        }

        showStatus(elements.dataStatus(), 'Loading and analyzing data...', 'loading');

        const trainText = await readFile(trainFile);
        console.log('File read successfully, length:', trainText.length);
        
        appState.rawData = parseCSV(trainText);
        console.log('Data parsed successfully, rows:', appState.rawData.length);
        
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
        console.error('Data loading error:', error);
        showStatus(elements.dataStatus(), `❌ Data loading error: ${error.message}`, 'error');
    }
};

window.preprocessData = function() {
    try {
        showStatus(elements.preprocessStatus(), 'Engineering features and preprocessing...', 'loading');
        
        // Extract features and targets
        const features = appState.rawData.map(row => {
            const featureRow = {};
            SCHEMA.features.forEach(feature => {
                featureRow[feature] = row[feature];
            });
            return featureRow;
        });
        
        const targets = appState.rawData.map(row => row[SCHEMA.target] ? 1 : 0);
        
        // For now, use simple preprocessing without data leakage fix
        const processedFeatures = imputeMissingValues(features);
        const engineeredFeatures = engineerNewFeatures(processedFeatures);
        
        const quantitativeData = engineeredFeatures.map(row => 
            [...SCHEMA.quantitative, ...Object.keys(SCHEMA.derivedFeatures)].map(feature => row[feature])
        );
        
        const { standardizedData, means, stds } = standardizeDataWithStats(quantitativeData);
        
        appState.processedData = {
            features: standardizedData,
            targets: targets,
            featureNames: [
                ...SCHEMA.quantitative,
                ...Object.keys(SCHEMA.derivedFeatures)
            ]
        };
        
        showStatus(elements.preprocessStatus(), 
            `✅ Preprocessing complete! ${standardizedData[0].length} features`, 'success');
        
        if (elements.featureInfo()) {
            elements.featureInfo().innerHTML = `
                <strong>🔧 Feature Engineering:</strong>
                <div style="margin-top: 8px;">
                    • Original features: ${SCHEMA.features.length}<br>
                    • Engineered features: ${Object.keys(SCHEMA.derivedFeatures).length}<br>
                    • Total features: ${standardizedData[0].length}<br>
                    • Samples: ${standardizedData.length}
                </div>
            `;
        }
        
        updateUIState();
        
    } catch (error) {
        showStatus(elements.preprocessStatus(), `❌ Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
};

window.imputeMissingValues = function(features) {
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
};

window.engineerNewFeatures = function(features) {
    return features.map(row => {
        const newRow = { ...row };
        Object.entries(SCHEMA.derivedFeatures).forEach(([name, fn]) => {
            newRow[name] = fn(row);
        });
        return newRow;
    });
};

window.standardizeDataWithStats = function(data) {
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
};

window.createModel = function() {
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
            metrics: ['accuracy']
        });
        
        const totalParams = appState.model.countParams();
        
        if (elements.modelSummary()) {
            elements.modelSummary().innerHTML = `
                <strong>🧠 Neural Network Architecture:</strong>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9rem; margin-top: 10px;">
                    <strong>Model Layers:</strong><br>
                    • Input: ${appState.processedData.featureNames.length} features<br>
                    • Hidden: Dense(12 units, ReLU activation)<br>
                    • Dropout: 30% rate<br>
                    • Output: Dense(1 unit, Sigmoid activation)<br>
                    • Total Parameters: ${totalParams.toLocaleString()}
                </div>
            `;
        }
        
        showStatus(elements.modelStatus(), '✅ Neural network created successfully!', 'success');
        updateUIState();
        
    } catch (error) {
        showStatus(elements.modelStatus(), `❌ Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Loan Approval AI Application Initialized');
    console.log('📊 Features:', SCHEMA.features);
    console.log('🎯 Target:', SCHEMA.target);
    
    // Test if functions are available
    console.log('loadData function available:', typeof window.loadData === 'function');
    console.log('readFile function available:', typeof window.readFile === 'function');
    
    // Set up initial UI state
    updateUIState();
});

// Add simple test function
window.testApp = function() {
    console.log('App test: All functions should be available');
    console.log('appState:', appState);
    console.log('SCHEMA:', SCHEMA);
    alert('App is loaded! Check console for details.');
};