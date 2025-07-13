% Select Dataset
dataset_name = ['breast_cancer']; % Change to 'iris', 'wine', or 'breast_cancer'

% Load The Dataset
%iris
if strcmp(dataset_name, 'iris')
    data = readtable('C:\Users\Marilou\Desktop\dataset\iris\iris.data', 'FileType', 'text');
    X = table2array(data(:, 1:4)); % Iris features
    labels = data{:, 5};
    [unique_labels, ~, y] = unique(labels); % Convert Iris labels to numeric
    n_clusters = 3; 

%wine
elseif strcmp(dataset_name, 'wine')
    data = readtable('C:\Users\Marilou\Desktop\dataset\wine\wine.data', 'FileType', 'text');
    X = table2array(data(:, 2:end)); % Wine features
    y = table2array(data(:, 1)); % Wine labels
    n_clusters = 3; 

%breast cancer
elseif strcmp(dataset_name, 'breast_cancer')
    
    data = readtable('C:\Users\Marilou\Desktop\dataset\breast+cancer+wisconsin+diagnostic\wdbc.data', 'FileType', 'text');
    
    % Extract the labels (Var2) and features (Var3 to the end)
    y = data.Var2; 

    % Convert labels to numeric 
    unique_labels = unique(y);
    y_numeric = zeros(size(y));

    for i = 1:length(unique_labels)
        y_numeric(ismember(y, unique_labels{i})) = i;
    end

    y = y_numeric;
    
    % Extract features
    X = table2array(data(:, 3:end)); % Breast Cancer features (excluding the first two columns)
    
    n_clusters = 3; 
    
else
    error('Invalid dataset name');
end

% Parameters for CKMeans
relaxation_parameter = 0.1; % Example parameter

ckmeans = CKMeans(n_clusters, relaxation_parameter);
ckmeans = ckmeans.fit(X, y);

predicted_labels = ckmeans.predict(X);

%Results
disp('Predicted Cluster Labels:');
disp(predicted_labels);

disp('True Labels vs Predicted Cluster Labels:');
disp(table(y, predicted_labels));

% Visualization
colors = 'rgb'; 
figure;
hold on;

if strcmp(dataset_name, 'iris')
    % Visualization for Iris dataset
    for i = 1:max(predicted_labels)
        scatter(X(predicted_labels == i, 1), X(predicted_labels == i, 2), 36, colors(i), 'filled');
    end
    xlabel('Sepal Length');
    ylabel('Sepal Width');
    title('CKMeans Clustering Results on Iris Dataset (Sepal Length vs Sepal Width)');

elseif strcmp(dataset_name, 'wine')
    % Visualization for Wine dataset
    for i = 1:max(predicted_labels)
        scatter(X(predicted_labels == i, 1), X(predicted_labels == i, 2), 36, colors(i), 'filled');
    end
    xlabel('Alcohol');
    ylabel('Malic Acid');
    title('CKMeans Clustering Results on Wine Dataset (Alcohol vs Malic Acid)');

elseif strcmp(dataset_name, 'breast_cancer')
    % Visualization for Breast Cancer dataset
    for i = 1:max(predicted_labels)
        scatter(X(predicted_labels == i, 1), X(predicted_labels == i, 2), 36, colors(i), 'filled');
    end
    % Replace 'Feature1' and 'Feature2' with actual feature names
    xlabel('Feature1');
    ylabel('Feature2');
    title('CKMeans Clustering Results on Breast Cancer Dataset (Feature1 vs Feature2)');
end

legend('Cluster 1', 'Cluster 2', 'Cluster 3');
hold off;

if exist('y', 'var')
    % Confusion Matrix
    unique_labels = unique([y; predicted_labels]);
    confusion_matrix = zeros(length(unique_labels));
    for i = 1:length(unique_labels)
        for j = 1:length(unique_labels)
            confusion_matrix(i,j) = sum((y == unique_labels(i)) & (predicted_labels == unique_labels(j)));
        end
    end

    disp('Confusion Matrix:');
    disp(confusion_matrix);

    % Accuracy 
    accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix, 'all');
    disp(['Accuracy: ', num2str(accuracy)]);
end
