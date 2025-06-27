// Re-export either the I/O or Web implementation
export 'download_helper_io.dart'
    if (dart.library.html) 'download_helper_web.dart';
