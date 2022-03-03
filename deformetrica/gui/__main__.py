import sys
import traceback


# from PyQt5 import QtCore
#
#
# sys._excepthook = sys.excepthook
#
#
# class ExceptionHandler(QtCore.QObject):
#     errorSignal = QtCore.pyqtSignal()
#
#     def __init__(self):
#         super(ExceptionHandler, self).__init__()
#
#     def handler(self, exctype, value, traceback):
#         self.errorSignal.emit()
#         sys._excepthook(exctype, value, traceback)
#
#
# exceptionHandler = ExceptionHandler()
# sys.excepthook = exceptionHandler.handler


def _log_all_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """Log all uncaught exceptions in non-interactive mode.

    All python exceptions are handled by function, stored in
    ``sys.excepthook.`` By rewriting the default implementation, we
    can modify handling of all uncaught exceptions.

    Warning: modified behaviour (logging of all uncaught exceptions)
    applies only when running in non-interactive mode.

    """
    # ignore KeyboardInterrupt
    # if not issubclass(exc_type, KeyboardInterrupt):
    #     ROOT_LOGGER.error("", exc_info=(exc_type, exc_value, exc_traceback))

    # logger.info('exc_type=' + str(exc_type) + ', exc_value=' + str(exc_value) + ', exc_traceback=' + str(exc_traceback))
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    return


sys.excepthook = _log_all_uncaught_exceptions


def main():
    import gui
    gui_main = gui.StartGui()
    gui_main.start()


if __name__ == '__main__':
    main()

