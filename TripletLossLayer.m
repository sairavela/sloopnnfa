classdef TripletLossLayer < nnet.layer.RegressionLayer
   
   properties
       
   end
    
   methods
       function layer = TripletLossLayer(name)
           if nargin == 1
               layer.Name = name;
           end
           
           layer.Description = 'Triplet Loss Layer';
       end
   
       function loss = forwardLoss(layer, Y, T) 
           Anchor = Y(:, :, :, 1:3:end);
           Positive = Y(:, :, :, 2:3:end);
           Negative = Y(:, :, :, 3:3:end);
           
           AnchorNorm = bsxfun(@rdivide, Anchor, sqrt(sum(Anchor.^2)));
           PositiveNorm = bsxfun(@rdivide, Positive, sqrt(sum(Positive.^2)));
           NegativeNorm = bsxfun(@rdivide, Negative, sqrt(sum(Negative.^2)));
           
           PositiveDiff = sqrt(sum(squeeze(AnchorNorm - PositiveNorm).^2));
           NegativeDiff = sqrt(sum(squeeze(AnchorNorm - NegativeNorm).^2));
           
           loss = (PositiveDiff - NegativeDiff) + 0.3;
           loss = mean(loss);
           loss = max(0, loss);
       end
       
       function dLdX = backwardLoss(layer, Y, T) 
           N = size(Y, 4);
           
           Anchor = Y(:, :, :, 1:3:end);
           Positive = Y(:, :, :, 2:3:end);
           Negative = Y(:, :, :, 3:3:end);
           
           DiffTriplets1 = sign(Negative - Positive);
           DiffTriplets2 = sign(Positive - Anchor);
           DiffTriplets3 = sign(Negative - Anchor);
           
	   try
           	DiffTriplets = gpuArray(zeros(size(DiffTriplets1).*([1 1 1 3])));
           	DiffTriplets(:, :, :, 1:size(DiffTriplets1, 4)) = DiffTriplets1;
           	DiffTriplets(:, :, :, (1:size(DiffTriplets1, 4)) + size(DiffTriplets1, 4)) = DiffTriplets2;
           	DiffTriplets(:, :, :, (1:size(DiffTriplets1, 4)) + 2 * size(DiffTriplets1, 4)) = DiffTriplets3;
           
	   catch ME
	   	fprintf(2, e.identifier);
		fprintf(2, e.message);
	   end
           dLdX = DiffTriplets/N;
           dLdX = single(dLdX);
       end
   end
   
end
