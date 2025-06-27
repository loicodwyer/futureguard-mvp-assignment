// lib/providers/scenario_state.dart
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/risk_result.dart';
import '../services/api_client.dart';

class ScenarioState extends ChangeNotifier {
  // sliders
  double incomeLoss = 0;
  double moreExpenses = 0;
  double investReturn = 0;
  double inflation = 0;

  // last result
  RiskResult? _result; // private
  RiskResult? get result => _result; // public

  // busy flag so the UI can show a spinner
  bool _busy = false;
  bool get busy => _busy;
  bool get hasResult => _result != null && !_busy;

  // ------------------------------------------------------------------
  // 1. Mock loader – used once at app start so the UI isn’t empty
  // ------------------------------------------------------------------

  Future<void> loadMock() async {
    final raw = await rootBundle.loadString('assets/mock_risk_result.json');
    _result = RiskResult.fromJson(jsonDecode(raw));
    notifyListeners();
  }

  // ------------------------------------------------------------------
  // 2. Fetch result from the FastAPI server
  // Now accepts userId, start, horizon
  // ------------------------------------------------------------------

  Future<void> fetchRiskResult({
    required int userId,
    required DateTime start,
    required int horizon,
    String baseUrl = 'http://127.0.0.1:8000',
  }) async {
    _busy = true;
    notifyListeners();

    try {
      final apiClient = ApiClient.instance..baseUrl = baseUrl;

      final data = await apiClient.runRisk(
        userId: userId,
        start: start,
        horizon: horizon,
      );

      _result = RiskResult.fromJson(data);
    } catch (e) {
      debugPrint('❌ fetchRiskResult failed: $e');
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  // ------------------------------------------------------------------
  // 3. Called by each slider
  // ------------------------------------------------------------------
  void update({
    double? incomeLoss,
    double? moreExpenses,
    double? investReturn,
    double? inflation,
  }) {
    this.incomeLoss = incomeLoss ?? this.incomeLoss;
    this.moreExpenses = moreExpenses ?? this.moreExpenses;
    this.investReturn = investReturn ?? this.investReturn;
    this.inflation = inflation ?? this.inflation;
    notifyListeners();
  }

  // Optional helper
  void setResult(RiskResult newResult) {
    _result = newResult;
    notifyListeners();
  }
}
