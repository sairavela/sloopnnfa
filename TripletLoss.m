classdef TripletLoss < nnet.layer.RegressionLayer
   
    properties
    
    end
    
    methods
       
        function layer = TripletLoss(name)
            if nargin == 1
                layer.Name = name;
            end
           
            layer.Description = 'Triplet Loss Layer';
        end
        
        function loss = forwardLoss(layer, Y, T)
            
            Anchor = Y(:, :, :, 1:3:end);
            Positive = Y(:, :, :, 2:3:end);
            Negative = Y(:, :, :, 3:3:end);
            
            N = size(Anchor, 4);

            PositiveDiff = Anchor - Positive;
            NegativeDiff = Anchor - Negative;
            
            loss = 0;
            
            for i = 1:N
                posVal = permute(PositiveDiff(:, :, :, i), [3 1 2]);
                posNorm = posVal' * posVal;
                negVal = permute(NegativeDiff(:, :, :, i), [3 1 2]);
                negNorm = negVal' * negVal;
                tempLoss = posNorm - negNorm + 0.2;
                if tempLoss > 0
                    loss = loss + (tempLoss / N);
                end 
            end

	    loss = gpuArray(single(loss));
        end
        
        function dLdX = backwardLoss(layer, Y, T)
            
            Anchor = Y(:, :, :, 1:3:end);
            Positive = Y(:, :, :, 2:3:end);
            Negative = Y(:, :, :, 3:3:end);
            
            N = size(Anchor, 4);

            PositiveDiff = Anchor - Positive;
            NegativeDiff = Anchor - Negative;
            PosNegDiff = Negative - Positive;
	    
	    try            
            	DiffTriplets = gpuArray(zeros(size(Y).*([1 1 1 1])));
            
            	for i = 1:N
                	posVal = permute(PositiveDiff(:, :, :, i), [3 1 2]);
                	posNorm = posVal' * posVal;
                	negVal = permute(NegativeDiff(:, :, :, i), [3 1 2]);
                	negNorm = negVal' * negVal;
                	loss = posNorm - negNorm + 0.2;
                	if loss >= 0
                    		DiffTriplets(:, :, :, 3 * i - 2) = 2 * PosNegDiff(:, :, :, i);
                    		DiffTriplets(:, :, :, 3 * i - 1) = -2 * PositiveDiff(:, :, :, i);
                    		DiffTriplets(:, :, :, 3 * i) = 2 * NegativeDiff(:, :, :, i);
                	end
            	end
            
            	DiffTriplets = DiffTriplets / N;
            
            	dLdX = single(DiffTriplets);

	   catch ME
		fprintf(2, ME.identifier);
	   	fprintf(2, ME.message);
           end 
        end
        
    end
    
end
