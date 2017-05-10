function getedge=edgesDemo(image)
% Demo for Structured Edge Detector (please see readme.txt first).

%addpath
addpath('.\matlab');
addpath('.\channels');
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
% tic, 
model=edgesTrain(opts);
% toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results

outputdir=dir(fullfile('../edge_input','*.jpg')); 
[a,b]=size(outputdir); 
%time=0;
for i=1:a 
     I=imread(fullfile('../edge_input',outputdir(i).name)); 

%tic
     disp( outputdir(i).name );
     E=edgesDetect(I,model);
% time=[time toc]
%      figure,imshow(I); 
%      figure,imshow(E);
%      I1=rgb2gray(I);
%      I1=histeq(I1,256); 
%      I2=edge(I1,'Sobel');
% %      figure,imshow(I2);
     
    
    
     imwrite(E,fullfile('../edge_output',[ strtok(outputdir(i).name,'.') '_edge.jpg']))
     disp( [i strtok(outputdir(i).name,'.') '_edge.jpg'] );
%      imwrite(I2,[int2str(i) '_sobel_edge.jpg'])
end
disp('Finish!');
% I = imread('alice03.jpg');
% I=image;
% tic, E=edgesDetect(I,model); 
% toc
% figure(1); imshow(I); figure(2); imshow(1-E);
% getedge=(1-E);
end
