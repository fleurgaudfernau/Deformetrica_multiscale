%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                      %
%                                     Deformetrica                                     %
%                                                                                      %
%    Copyright Inria and the University of Utah.  All rights reserved. This file is    %
%    distributed under the terms of the Inria Non-Commercial License Agreement.        %
%                                                                                      %
%                                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function []=VTKPolyDataWriter(Pts, Poly, Scalars, Colors, TextureCoordinates, filename);

% Usage:
% VTKPolyDataWriter(Pts, Tri, Scalars, Colors, TextureCoordinates, filename);

  fid = fopen(filename, 'w');
  
	nPts = size(Pts,1);
	Dim = size(Pts,2);
	if (Dim~=3)
		disp(['Error: points in 3D only, not ' num2str(Dim) 'D!']);
	end
	
  fprintf(fid, '# vtk DataFile Version 3.0\nvtk output\nASCII\n');
  fprintf(fid, 'DATASET POLYDATA\n');
  fprintf(fid, 'POINTS %d float\n', nPts);
  
  for i=1:nPts
		fprintf(fid, '%f %f %f\n',Pts(i,1),Pts(i,2),Pts(i,3));
  end
  
	if ~isempty(Poly) % write the polygons if any
		nPoly = size(Poly,1);
		PolyDim = size(Poly,2);
		if (PolyDim~=2)&&(PolyDim~=3)
			disp('Only segments and triangles are implemented as polygons')
		end
		
		fprintf(fid, 'POLYGONS %d %d\n', nPoly, (PolyDim + 1)*nPoly);
		for i=1:nPoly
			switch PolyDim
			case 2
				fprintf(fid,'%d %d %d\n',2,Poly(i,1)-1,Poly(i,2)-1);
			case 3
		    fprintf(fid,'%d %d %d %d\n',3,Poly(i,1)-1,Poly(i,2)-1,Poly(i,3)-1);
		  end
		end
		
		fprintf(fid,'\nCELL_DATA %d\n',nPoly);
	end
	
	fprintf(fid,'POINT_DATA %d\n',nPts);

	if( ~isempty(Scalars) ) % write the scalars if any
		
		if (length(Scalars)~=nPts)
			disp('Warning: not point data!');
		end

		fprintf(fid,'SCALARS scalars double\n');
    fprintf(fid, 'LOOKUP_TABLE default\n');
    for i=1:length(Scalars)
    	fprintf(fid, '%f\n',Scalars(i));
		end      
	end
	
	if (~isempty(Colors)) % write colors if any
		
		if (size(Colors,1)~=nPts)
			disp('Warning: nb of columns in colors array does not match number of points!');
		end
		
		nValues = size(Colors,2);
				
		% fprintf(fid,'POINT_DATA %d\n',nPts);
		fprintf(fid,'COLOR_SCALARS colors %d\n', nValues);
		for i=1:nPts
			for j=1:nValues
				fprintf(fid,'%d ', Colors(i,j));
			end
			fprintf(fid,'\n');
		end
		
	end
	
	if (~isempty(TextureCoordinates)) % write colors if any
		
		if (size(TextureCoordinates,1)~=nPts)
			disp('Warning: nb of columns in TextureCoordinates array does not match number of points!');
		end
		
		nValues = size(TextureCoordinates,2);
				
		% fprintf(fid,'POINT_DATA %d\n',nPts);
		fprintf(fid,'TEXTURE_COORDINATES TextureCoordinates %d float\n', nValues);
		for i=1:nPts
			for j=1:nValues
				fprintf(fid,'%f ', TextureCoordinates(i,j));
			end
			fprintf(fid,'\n');
		end
		
	end

		  
  fclose(fid);

end