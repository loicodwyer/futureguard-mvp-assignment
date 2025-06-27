// lib/providers/overview_state.dart

import 'package:flutter/material.dart';
import '../models/account_overview.dart';
import '../services/api_client.dart';

class OverviewState extends ChangeNotifier {
  // User selections
  int userId = 10;
  DateTime start = DateTime(2025, 3, 1);
  int horizon = 92; // days into the future

  // Response data
  AccountOverview? data;
  bool busy = false;
  String? error;

  // Fetch from backend
  Future<void> fetch() async {
    busy = true;
    error = null;
    notifyListeners();

    try {
      final raw = await ApiClient.instance.accountOverview(
        userId: userId,
        start: start,
        horizon: horizon,
      );
      data = AccountOverview.fromJson(raw);
    } catch (e) {
      error = '⚠️ ${e.toString()}';
      data = null;
    } finally {
      busy = false;
      notifyListeners();
    }
  }

  // For changing user selections
  void update({int? userId, DateTime? start, int? horizon}) {
    if (userId != null) this.userId = userId;
    if (start != null) this.start = start;
    if (horizon != null) this.horizon = horizon;
    notifyListeners();
  }
}
