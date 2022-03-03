import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

#from vtk import vtkPolyDataReader, vtkPolyDataWriter, vtkPolyDataNormals

if __name__ == '__main__':

    """
    Basic info printing.
    """

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')
    logger.info('')

    """
    Read command line.
    """

    assert len(sys.argv) in [2, 3], 'Usage: ' + sys.argv[0] + " <path/to/source.vtk> [path/to/output.vtk (optional)] "

    source_path = sys.argv[1]

    if len(sys.argv) == 2: output_path = source_path
    else: output_path = sys.argv[2]

    if not os.path.isfile(source_path):
        raise RuntimeError('The specified source file ' + source_path + ' does not exist.')

    """
    Core part.
    """

    number_of_iterations = 0

    # Reading source file.
    reader = vtkPolyDataReader()
    reader.SetFileName(source_path)
    reader.Update()
    input = reader.GetOutput()

    # Reorienting.
    normalf = vtkPolyDataNormals()
    normalf.SetInputData(input)
    normalf.ConsistencyOn()
    normalf.AutoOrientNormalsOn()  # Should have closed surface
    normalf.FlipNormalsOn()
    normalf.Update()

    output = normalf.GetOutput()
    output.BuildLinks()

    # Saving results.
    writer = vtkPolyDataWriter()
    writer.SetInputData(output)
    writer.SetFileName(output_path)
    writer.Update()
