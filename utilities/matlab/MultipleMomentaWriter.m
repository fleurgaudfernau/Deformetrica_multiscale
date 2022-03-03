function [] = MultipleMomentaWriter(PtsFilename, MomFilename, outFilename);

CP = load(PtsFilename);
MOM = readMomentaFile(MomFilename);

MOM = permute(MOM,[2 1 3]);
nsub = size(MOM,3);


if (~isequal(size(CP),size(MOM(:,:,1))))
    disp('error: dimension mismatch');
    return;
end


nPts = size(CP,1);
Dim = size(CP,2);
if (Dim~=3)
    disp(['Error: points in 3D only, not ' num2str(Dim) 'D!']);
    return;
end

fid = fopen(outFilename,'w');

fprintf(fid, '# vtk DataFile Version 3.0\nvtk output\nASCII\n');
fprintf(fid, 'DATASET POLYDATA\n');
fprintf(fid, 'POINTS %d float\n', nsub*nPts);

for su = 1:nsub
    
    for i=1:nPts
        fprintf(fid, '%f %f %f\n',CP(i,1),CP(i,2),CP(i,3));
    end
    
end

fprintf(fid,'\nPOINT_DATA %d\n',nsub*nPts);
fprintf(fid,'VECTORS Momenta float\n');

for su = 1:nsub
    
    for i=1:nPts
        fprintf(fid, '%f %f %f\n',MOM(i,1,su),MOM(i,2,su),MOM(i,3,su));
    end
    
end


fprintf(fid,'SCALARS diagnosis double\n');
fprintf(fid,'LOOKUP_TABLE default\n');

for su = 1:(nsub)
    
    for i=1:nPts
        fprintf(fid, '%f\n',su);
    end
    
end


fclose(fid);

end
