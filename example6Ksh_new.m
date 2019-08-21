close all;
clear all;

addpath(genpath('/net/fog/home/bakliwal/'));

outputFolder = fullfile('geckodata');
rootFolder = fullfile(outputFolder, 'patches');

indices = 1:251;

imds = imageDatastore(fullfile(rootFolder), 'IncludeSubFolders', true, 'LabelSource', 'foldernames', 'FileExtensions', {'.mat', '.jpg'}, 'ReadFcn', @customReadFcn);
imgStore = subset(imds, indices);

[trainingSet, validationSet] = splitEachLabel(imgStore, 2);

%%

lapFilter = fspecial('laplacian', 0.2);
filter = [-1 -1 -1; -1 8 -1; -1 -1 -1];

displacements = zeros([251 251]);
qxs = zeros(64, 64, 251, 251);
qys = zeros(64, 64, 251, 251);
divs = zeros(64, 64, 251, 251);
divVals = zeros(251, 251);

images = zeros(64, 64, 3, 251);
%%
parfor i = 1:251
	disp(i);
    
	temp = readimage(imgStore, i);
    	summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    	temp(:, :, 1) = (temp(:, :, 1) ./ summ);
    	temp(:, :, 2) = (temp(:, :, 2) ./ summ);
    	temp(:, :, 3) = (temp(:, :, 3) ./ summ);
   	resImage = temp;
    	images(:, :, :, i) = imresize(resImage, [64 64]);
    
    	for j = 1:251
        
    		img1 = resImage;
        	temp1 = rgb2gray(img1);
        	A = imresize(temp1, [64 64]);

		temp = readimage(imgStore, j);
    		summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    		temp(:, :, 1) = (temp(:, :, 1) ./ summ);
    		temp(:, :, 2) = (temp(:, :, 2) ./ summ);
    		temp(:, :, 3) = (temp(:, :, 3) ./ summ);
    		img2 = temp;
        	%img2 = getNormalized(imgStore, j);
        	temp2 = rgb2gray(img2);
        	F = imresize(temp2, [64 64]);

        	X = A'; Y = F';
%         	X = (X-min(X(:)))./(max(X(:)-min(X(:))))*255;
%         	Y = (Y-min(Y(:)))./(max(Y(:))-min(Y(:)))*255;
        	[qx,qy] = FA2DImNoH(X,Y,64,1,2^12,2,0);
        	qxs(:, :, i, j) = qx;
        	qys(:, :, i, j) = qy;
        	divs(:, :, i, j) = divergence(qx, qy);
        	displacements(i, j) = sum(abs(qx(:))) + sum(abs(qy(:)));
        
        	image1 = imresize(img1, [64 64]);
        	image2 = imresize(img2, [64 64]);

       		m1 = min(image1, [], 3);
        	sm1 = sum(m1,2);
        	lvec = 1:length(sm1);
        	lvec = lvec(:);
        	sm1=abs(sm1-polyval(polyfit(lvec,sm1,1),lvec));
        	sm1 = sm1./sum(sm1);
        	mpos1 = sum(lvec(:).*sm1);
        	spos1 = sqrt(sum((lvec(:)-mpos1).^2.*sm1));

%         	subplot(221);plot(sm1)
%         	subplot(222);imagesc(image1);hold on;
%         	line([1 size(m1,2)],[mpos1-spos1 mpos1-spos1]);
%         	line([1 size(m1,2)],[mpos1+spos1 mpos1+spos1]);

%         	hold off;
        	m2 = min(image2, [], 3);
        	sm2 = sum(m2,2);
        	lvec = 1:length(sm2);
        	lvec = lvec(:);
        	sm2=abs(sm2-polyval(polyfit(lvec,sm2,1),lvec));
        	sm2 = sm2./sum(sm2);
        	mpos2 = sum(lvec(:).*sm2);
        	spos2 = sqrt(sum((lvec(:)-mpos2).^2.*sm2));

%         	subplot(223);plot(sm2)
%         	subplot(224);imagesc(image2);hold on;
%         	line([1 size(m2,2)],[mpos2-spos2 mpos2-spos2]);
%         	line([1 size(m2,2)],[mpos2+spos2 mpos2+spos2]);
% 
%         	hold off;

        	wvec = sm1+sm2;
        	wvec = wvec./sum(wvec);
        	div = divs(:, :, i, j);
        	divVals(i, j) = sum(sum(abs(repmat(wvec,[1 size(div,2)]).*div)));
	end
end

save('dispWorkspace6');

%ss = sprintf('diffs/example2.png');
%print(gcf,'-dpng',ss);

%function data = customReadFcn(filename)
%	newImg = load(filename);
 %   	data = newImg.icolor;
%end

%{
function resImage = getNormalized(trainingSet, index)
    	temp = readimage(trainingSet, index);
    	summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
    	temp(:, :, 1) = temp(:, :, 1) ./ summ;
    	temp(:, :, 2) = temp(:, :, 2) ./ summ;
    	temp(:, :, 3) = temp(:, :, 3) ./ summ;
    	resImage = temp;
end
%}
