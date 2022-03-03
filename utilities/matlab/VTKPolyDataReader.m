%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                      %
%                                     Deformetrica                                     %
%                                                                                      %
%    Copyright Inria and the University of Utah.  All rights reserved. This file is    %
%    distributed under the terms of the Inria Non-Commercial License Agreement.        %
%                                                                                      %
%                                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Points, Tri, Scalars, Colors, TextureCoordinates] = VTKPolyDataReader(filename)

% Usage:
% [Points, Tri, Scalars, Colors, TextureCoordinates] = VTKPolyDataReader(filename)


fid = fopen(filename, 'r');
if(fid==-1)
    fprintf(1,'Error: file descriptor not valid, check the file name.\n');
    return;
end


keyWord = 'DATASET POLYDATA';
newL = GoToKeyWord(fid, keyWord);
if(newL == -1)
    fprintf(1, 'Error: file is not a vtkPolyData.\n');
    return;
end


% Output:
Points = [];  % set of vertices
Tri = [];     % set of triangles
Scalars = []; % set of scalars
Colors = [];  % Colors
TextureCoordinates = []; % TextureCoordinates


newL = fgetl(fid);
keyWord = 'POINTS';
newL = GoToKeyWord(fid, keyWord);
if(newL==-1)
    fprintf(1, 'Cannot find flag: %s\n', keyWord);
end

buffer = sscanf(newL,'%s%d%s');
numPoints = buffer(length(keyWord)+1); % because these are points

newL = fgetl(fid);
count = 1;

% Read the points data
while(count<=3*numPoints)
    [num,c] = sscanf(newL,'%f');
    Points = [Points;num];
    
    count = count + c;
    newL = fgetl(fid);
end
Points = reshape(Points, [3,numPoints])';
% end of point data



% Read the polygons
keyWord = 'POLYGONS';
newL = GoToKeyWord(fid, keyWord);
if(newL == -1)
    keyWord = 'LINES';
    newL = GoToKeyWord(fid, keyWord);
    if (newL == -1)
        return;
    end
end
buffer = sscanf(newL, '%s%d%d');
numPoly = buffer(length(keyWord)+1); % get the number of polygons
numTotal = buffer(length(keyWord)+2); % get the actual number of things to read

dimPoly = numTotal / numPoly;
if (dimPoly==4)
	% disp('polygons detected as triangles');
	
	Tri = zeros(numPoly,3);

	for i=1:numPoly
    	newL = fgetl(fid);
    	buffer = sscanf(newL, '%d %d %d %d'); % only triangles are supported
    
    	Tri(i,:) = buffer(2:4)';
        
	end
	Tri = Tri+1; % problem with index in matlab / c
elseif (dimPoly==3)
	% disp('polygons detected as edges');
	
	Tri = zeros(numPoly,2);
	for i=1:numPoly
    	newL = fgetl(fid);
    	buffer = sscanf(newL, '%d %d %d'); % only triangles are supported
    
    	Tri(i,:) = buffer(2:3)';
        
	end
	Tri = Tri + 1;
else
	disp('cannot determine which polygons are used');
	return;
end

% end of polygons


keyWord = 'CELL_DATA';
newL = GoToKeyWord(fid, keyWord);
if(newL == -1)
    return;
end
buffer = sscanf(newL, '%s%d');
numCellData = buffer(length(keyWord)+1);

% keyWord = 'NORMALS';
% newL = GoToKeyWord(fid, keyWord);
% if(newL == -1)
%     fprintf(1, 'No normal data\n');
% else
%       
%     count = 1;
%     while(count <= 3*numCellData)
%         
%         newL = fgetl(fid);
%         [buffer,c] = sscanf(newL, '%f');
%         count = count+c;
%         Normals = [Normals; buffer];
%         
%     end
%    
%     Normals = reshape(Normals, [3, numCellData])';
%     
%     
% end

keyWord = 'POINT_DATA';
newL = GoToKeyWord(fid, keyWord);
if(newL ==-1)
    return;
end
buffer = sscanf(newL,'%s%d');
numPointData = buffer(length(keyWord)+1);



% Read the scalars
keyWord = 'SCALARS';
newL = GoToKeyWord(fid, keyWord);
if(newL == -1)
    fprintf(1, 'No scalar\n');
else
        
    keyWord = 'LOOKUP_TABLE';
    newL = GoToKeyWord(fid, keyWord);
    if(newL == -1)
        fprintf(1, 'No LUT\n');
        keyWord = 'SCALARS';
        newL = GoToKeyWord(fid, keyWord);
    end
    
    count = 1;
    while(count <= numPoints)
        
        newL = fgetl(fid);
        [buffer, c] = sscanf(newL, '%f');
        count = count + c;
        Scalars = [Scalars;buffer];
        
    end
    
end
% end of scalars



% Read the LUT: supposed to be right after the scalars!!
% keyWord = 'LOOKUP_TABLE';
% while(strncmp(newL, keyWord, length(keyWord))==0 & newL~=-1)
%     newL = fgetl(fid);
% end
% if(newL == -1)
%     fprintf(1,'There is no LUT.\n');
% else
%     
%     buffer = sscanf(newL, '%s%s%d');
%     numLUT = buffer(end);
%     
%     count = 1;
%     while(count <= 4*numLUT)
%         
%         newL = fgetl(fid);
%         [buffer, c] = sscanf(newL, '%f');
%         count = count + c;
%         LUT = [LUT; buffer];
%         
%     end
%    
%     LUT = reshape(LUT, [4,numLUT])';
%     
% end

% end of LUT

% Be careful: assume a set of coordinates per line (number of lines = number of points)
keyWord = 'COLOR_SCALARS';
newL = GoToKeyWord(fid, keyWord);
if(newL == -1)
    fprintf(1, 'No color_scalars\n');
else
    count = 1;

    while(count <= numPoints)
        
        newL = fgetl(fid);
        buffer = sscanf(newL, '%d');
        count = count + 1;
        Colors = [Colors;buffer'];
        
    end
    
end




% Be careful: assume a set of coordinates per line (number of lines = number of points)
% keyWord = 'TEXTURE_COORDINATES';
% newL = GoToKeyWord(fid, keyWord);
% if(newL == -1)
%     fprintf(1, 'No Texture Coordinates\n');
% else
%     count = 1;
% 
%     while(count <= numPoints)
%         
%         newL = fgetl(fid);
%         [buffer, c] = sscanf(newL, '%f');
%         count = count + 1;
%         TextureCoordinates = [TextureCoordinates;buffer];
%         
%     end
%     
% end


fclose(fid);

end
