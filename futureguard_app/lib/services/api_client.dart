// lib/services/api_client.dart
//
// Pure-Dart REST client for the FutureGuard FastAPI backend.
// All platform-specific download logic lives in helpers/download_helper_*.
// ---------------------------------------------------------------------

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import '../helpers/download_helper.dart' as dl; // ← handles file saving

class ApiClient {
  ApiClient._();
  static final instance = ApiClient._();

  /// Change this if FastAPI runs on a different host/port
  String baseUrl = 'http://127.0.0.1:8000';

  // ---------------------------------------------------------------------
  // 1. Ping - make sure the backend is alive
  // ---------------------------------------------------------------------

  Future<void> ping() async {
    final r = await http.get(Uri.parse('$baseUrl/ping'));
    if (r.statusCode != 200 || jsonDecode(r.body)['ok'] != true) {
      throw Exception('❌ API not reachable – status ${r.statusCode}');
    }
  }

  // ---------------------------------------------------------------------
  // 2. Run risk analysis
  // POST /run_risk_analysis/{userId}?start=YYYY-MM-DD&horizon=XX
  // ---------------------------------------------------------------------

  Future<Map<String, dynamic>> runRisk({
    required int userId,
    required DateTime start,
    required int horizon,
  }) async {
    final s =
        '${start.year}-${start.month.toString().padLeft(2, '0')}-${start.day.toString().padLeft(2, '0')}';
    final uri = Uri.parse(
      '$baseUrl/run_risk_analysis/$userId?start=$s&horizon=$horizon',
    );

    final r = await http.post(uri);
    if (r.statusCode != 200) {
      throw Exception('❌ Pipeline failed – ${r.body}');
    }
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  // ---------------------------------------------------------------------
  // 3. Account overview
  // GET /account_overview/{userId}?start=YYYY-MM-DD&horizon=XX
  // ---------------------------------------------------------------------

  Future<Map<String, dynamic>> accountOverview({
    required int userId,
    required DateTime start,
    required int horizon,
  }) async {
    final s =
        '${start.year}-${start.month.toString().padLeft(2, '0')}-${start.day.toString().padLeft(2, '0')}';
    final uri = Uri.parse(
      '$baseUrl/account_overview/$userId?start=$s&horizon=$horizon',
    );

    final r = await http.get(uri);
    if (r.statusCode != 200) {
      throw Exception('❌ Overview failed – ${r.body}');
    }
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  // ---------------------------------------------------------------------
  // 4. Download ZIP  (PNG + CSV)
  // POST /download_risk_analysis/{userId}?start=YYYY-MM-DD&horizon=XX
  // ---------------------------------------------------------------------

  Future<void> downloadRiskAnalysis({
    required int userId,
    required DateTime start,
    required int horizon,
  }) async {
    final s =
        '${start.year}-${start.month.toString().padLeft(2, '0')}-${start.day.toString().padLeft(2, '0')}';
    final uri = Uri.parse(
      '$baseUrl/download_risk_analysis/$userId?start=$s&horizon=$horizon',
    );

    final r = await http.post(uri);
    if (r.statusCode != 200) {
      throw Exception('❌ Download failed – ${r.body}');
    }

    // platform-specific write
    await dl.saveFile(
      Uint8List.fromList(r.bodyBytes),
      'risk_analysis_$userId.zip',
    );
  }
}
