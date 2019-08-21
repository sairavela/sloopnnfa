function resImage = getNormalized(trainingSet, index)
	temp = readimage(trainingSet, index);
    	summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    	temp(:, :, 1) = (temp(:, :, 1) ./ summ);
    	temp(:, :, 2) = (temp(:, :, 2) ./ summ);
    	temp(:, :, 3) = (temp(:, :, 3) ./ summ);
    	resImage = temp;
end
