 % Read the CT image

[filename_image,pathname_image] = uigetfile('*.*','Select the image file');

fid_image = fullfile(pathname_image,filename_image);
originalImage = imread(fid_image); 
size(originalImage)
disp(filename_image)
org_image=originalImage;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FFT enhancement 

% Perform FFT to obtain the frequency domain representation
fftImage = fftshift(fft2(originalImage)); 
enhancedFFTImage = fftImage * 1.2; 

% Perform the inverse FFT to obtain the enhanced image in the spatial domain
enhancedImage = ifft2(ifftshift(enhancedFFTImage));

enhancedImage = uint8(real(enhancedImage));
originalImage=enhancedImage;

if ndims(originalImage) == 2 || (ndims(originalImage) == 3 && size(originalImage, 3) == 1)

else
grayImage = rgb2gray(originalImage); 
end
filteredImage = medfilt2(grayImage, [5, 5]); 
contrastAdjusted = imadjust(filteredImage); 

% Thresholding to isolate the bright regions (assumed tumor) within the brain
binaryImage = contrastAdjusted < 187; % Adjust the threshold value to segment the bright regions

% Clean up the binary image using morphological operations
se = strel('disk', 6); % Define a disk-shaped structuring element
binaryImage = imopen(binaryImage, se); % Opening operation to remove small noise
binaryImage = bwareaopen(binaryImage, 1000); % Remove small objects

% Find the boundary of the bright region within the brain
boundary = bwboundaries(binaryImage, 'noholes');

% Display the original image and mark the boundary of the detected region
imshow(org_image);title(strcat('Filename:',filename_image));
hold on;
for k = 1 : length(boundary)
    b = boundary{k};
    plot(b(:, 2), b(:, 1), 'r', 'LineWidth', 2);
end
hold off;

% Write the boundary coordinates to a text file
[~,name,~]=fileparts(filename_image);

fileID = fopen(strcat(pathname_image,name,'_boundary_coordinates.txt'), 'w');
for k = 1 : length(boundary)
    b = boundary{k};
    % Writing boundary coordinates to the file
%     fprintf(fileID, 'Boundary %d:\n', k);

    for i = 1:size(b, 1)
        fprintf(fileID, '%d %d \r\n', b(i, 2), b(i, 1));
    end
    fprintf(fileID, '\n');
end
fclose(fileID);
disp(strcat('Boundary values are written to file: ',name,'_boundary_coordinates.txt'));
