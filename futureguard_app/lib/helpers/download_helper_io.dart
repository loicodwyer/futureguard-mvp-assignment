// Mobile / desktop – save to an app-documents folder
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

/// Saves [bytes] to a local file named [filename] and returns the path.
Future<String> saveFile(Uint8List bytes, String filename) async {
  final dir = await getApplicationDocumentsDirectory();
  final file = File(p.join(dir.path, filename));
  await file.writeAsBytes(bytes, flush: true);
  return file.path;
}
