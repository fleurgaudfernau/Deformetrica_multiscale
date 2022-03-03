function [] = MomentaWriter(PtsFilename, MomFilename, outFilename);

	CP = load(PtsFilename);
	MOM = load(MomFilename);
	
	if (~isequal(size(CP),size(MOM)))
		disp('error: dimension mismatch');
		return;
	end
	

	nPts = size(CP,1);
	Dim = size(CP,2);
	if (Dim~=3)&&(Dim~=2)
		disp(['Error: points in 2D or 3D only, not ' num2str(Dim) 'D!']);
		return;
	end
	
	fid = fopen(outFilename,'w');
	
  fprintf(fid, '# vtk DataFile Version 3.0\nvtk output\nASCII\n');
  fprintf(fid, 'DATASET POLYDATA\n');
  fprintf(fid, 'POINTS %d float\n', nPts);
  
	if (Dim==3)
	  for i=1:nPts
			fprintf(fid, '%f %f %f\n',CP(i,1),CP(i,2),CP(i,3));
	  end
	  
		fprintf(fid,'\nPOINT_DATA %d\n',nPts);
		fprintf(fid,'VECTORS Momenta float\n');
	  for i=1:nPts
	   	fprintf(fid, '%f %f %f\n',MOM(i,1),MOM(i,2),MOM(i,3));
		end
	end
	
	if (Dim==2)
	  for i=1:nPts
			fprintf(fid, '%f %f 0.0\n',CP(i,1),CP(i,2));
	  end

		fprintf(fid,'\nPOINT_DATA %d\n',nPts);
		fprintf(fid,'VECTORS Momenta float\n');
	  for i=1:nPts
	   	fprintf(fid, '%f %f 0.0\n',MOM(i,1),MOM(i,2));
		end
	end
		
		  
  fclose(fid);

end