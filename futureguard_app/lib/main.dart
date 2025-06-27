// lib/main.dart

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/scenario_state.dart';
import 'providers/overview_state.dart';
import 'screens/root_nav.dart';
import 'widgets/app_frame.dart';

void main() => runApp(const FutureGuardApp());

class FutureGuardApp extends StatelessWidget {
  const FutureGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // Loads static mock data for the Risk Dashboard tab
        ChangeNotifierProvider(create: (_) => ScenarioState()..loadMock()),

        // Fetches dynamic data from the FastAPI backend
        ChangeNotifierProvider(create: (_) => OverviewState()..fetch()),
      ],
      child: MaterialApp(
        builder:
            (context, child) => // <─ NEW: wrap once
                AppFrame(child: child ?? const SizedBox()),
        title: 'FutureGuard',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.orange),
        home: const RootNav(), // Bottom-tab layout controller
      ),
    );
  }
}
