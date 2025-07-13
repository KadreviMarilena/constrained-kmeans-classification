classdef CKMeans %DEFINE CLASS
    properties
        NClusters;
        RelaxationParameter;
        Hypercubes;
        Weights;
    end

    methods
        % arxikopoiei ena adikeimeno ckmeans me kayhorismeno arithmo clusters kai size hypercubes
        function obj = CKMeans(n_clusters, relaxation_parameter)
            obj.NClusters = n_clusters;
            obj.RelaxationParameter = relaxation_parameter;
            %arxikopoihsh sto nun
            obj.Hypercubes = cell(1, n_clusters);
            obj.Weights = cell(1, n_clusters);
        end

        %FIT METHOD
        %ipologismos oriwn kai varwn hypercube based in train data (x,y)
		%epanalambanete gia kathe klasi, ypologizei ton meso oro kai thn typikh apoklish gia kathe xaraktiristiko ta opoia xrisimopoioude gia ton upologismo ton orion kai tvn barvn
        function obj = fit(obj, X, y)
            classes = unique(y);
            [n_samples, n_features] = size(X);

            for i = 1:length(classes)
                c = classes(i);
                class_data = X(y == c, :);%ejagei to uposunolo tou x pou anikei sthn klasi c
                avg_ac = mean(class_data);%ypologismos mesis timis
                s_ac = std(class_data);%ypologismos typikhs apoklishs
                %kathorizei ta elaxista oria
                b_min_ac = avg_ac - obj.RelaxationParameter * s_ac;
                b_max_ac = avg_ac + obj.RelaxationParameter * s_ac;
                obj.Hypercubes{c} = [b_min_ac; b_max_ac];

                weights_c = 1 - (s_ac ./ (b_max_ac - b_min_ac));
                obj.Weights{c} = weights_c ./ sum(weights_c); % Normalized weights
            end
        end

        %PREDEICT METHOD
        % arxikopoiei ta kentra ton kivwn edos periorismou
        function labels = predict(obj, X)
            [n_samples, ~] = size(X);%kathorismos megethos train data
            labels = zeros(n_samples, 1);%dimiourgia pinaka etiketwn gia apothikefsh predict label

            % Assign each sample to the nearest centroid
            for i = 1:n_samples
                min_dist = inf;
                for j = 1:obj.NClusters
                    if ~isempty(obj.Hypercubes{j})
                        dist = obj.weightedEuclidean(X(i, :), obj.Hypercubes{j}(1, :), obj.Weights{j});
                        if dist < min_dist
                            min_dist = dist;
                            labels(i) = j;
                        end
                    end
                end
            end
       end

        %ypologizei thn efklidia apostash enos shmeiou dedomenvn kai enos kedrou
        function dist = weightedEuclidean(obj, point, centroid, weights)
            %ypologismos tis stathmismenis eykleidias apostashs
            dist = sqrt(sum(weights .* (point - centroid) .^ 2));
        end
        % Update centroids based on assigned data points
        function obj = updateCentroids(obj, X, labels)
            for j = 1:obj.NClusters
                if ~isempty(obj.Hypercubes{j})
                    cluster_points = X(labels == j, :);
                    obj.Centroids{j} = mean(cluster_points);
                end
            end
        end
    end
end
