load('dispWorkspace6');

momentVals = zeros(251, 251);

parfor i = 1:251
    for j = 1:251
        qx = qxs(:, :, i, j);
        qy = qys(:, :, i, j);
        
        temp = readimage(imgStore, i);
        summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
        temp(:, :, 1) = (temp(:, :, 1) ./ summ);
        temp(:, :, 2) = (temp(:, :, 2) ./ summ);
        temp(:, :, 3) = (temp(:, :, 3) ./ summ);
        img1 = temp;
        
        temp = readimage(imgStore, j);
        summ = temp(:, :, 1) + temp(:, :, 2) + temp(:, :, 3);
        temp(:, :, 1) = (temp(:, :, 1) ./ summ);
        temp(:, :, 2) = (temp(:, :, 2) ./ summ);
        temp(:, :, 3) = (temp(:, :, 3) ./ summ);
        img2 = temp;
        
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
        
        m2 = min(image2, [], 3);
        sm2 = sum(m2,2);
        lvec = 1:length(sm2);
        lvec = lvec(:);
        sm2=abs(sm2-polyval(polyfit(lvec,sm2,1),lvec));
        sm2 = sm2./sum(sm2);
        mpos2 = sum(lvec(:).*sm2);
        spos2 = sqrt(sum((lvec(:)-mpos2).^2.*sm2));
        
        wvec = sm1+sm2;
        wvec = wvec./sum(wvec);
	wvec = 1 - wvec;	
        
        crl = divergence(qx, qy);
        momentVals(i, j) = sum(sum(abs(repmat(wvec,[1 size(crl,2)]).*crl)));
    end
end

imgLabels = imgStore.Labels;
moments = zeros(251, 251);
for i = 1:251
    moments(i, :) = mean([momentVals(i, :); momentVals(:, i)']);
end
accuracy = 0;
momentLabels = zeros(251, 251);
for i = 1:251   
        [vals, idxs] = mink(moments(i, :), 2);
        momentLabels(i, idxs(2)) = 1;
        if imgLabels(idxs(2)) == imgLabels(i)
                accuracy = accuracy + 1;
        end
end

saveas(imagesc(momentLabels), 'momentLabels_test.png');
save('dispWorkspace_temp');
