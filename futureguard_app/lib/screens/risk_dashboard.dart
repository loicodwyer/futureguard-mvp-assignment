import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/scenario_state.dart';
import '../providers/overview_state.dart';
import '../widgets/fan_chart.dart';
import '../widgets/insight_block.dart';
import '../widgets/recommendations_block.dart';
import '../services/api_client.dart';

class RiskDashboard extends StatelessWidget {
  const RiskDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    final st = context.watch<ScenarioState>();
    final ov = context.watch<OverviewState>();
    final api = ApiClient.instance;

    // 1. ANALYSIS IS RUNNING
    if (st.busy) {
      return SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Text(
                'We are working on your personalised\n'
                'AI-driven Risk Analysis.\n'
                'We will let you know once the analysis is ready.',
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 16),
              CircularProgressIndicator(),
            ],
          ),
        ),
      );
    }

    // 2. NO RESULT YET - show centred “Run analysis” button
    if (!st.hasResult) {
      return SafeArea(
        child: Center(
          child: ElevatedButton.icon(
            icon: const Icon(Icons.refresh),
            label: const Text('Run analysis'),
            onPressed: () async {
              final sm = ScaffoldMessenger.of(context);
              sm.showSnackBar(
                const SnackBar(
                  content: Text(
                    'We are working on your personalised AI-driven Risk Analysis.\n'
                    'We will let you know once the analysis is ready.',
                  ),
                  duration: Duration(seconds: 3),
                ),
              );

              await context.read<ScenarioState>().fetchRiskResult(
                userId: ov.userId,
                start: ov.start,
                horizon: ov.horizon,
              );

              if (!context.mounted) return;
              sm.showSnackBar(
                const SnackBar(
                  content: Text('✅ Risk analysis has been finalised!'),
                  duration: Duration(seconds: 3),
                ),
              );
            },
          ),
        ),
      );
    }

    // 3. RESULT PRESENT - full scrollable dashboard
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Risk Dashboard',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),

            // run-again button (now centred at the top)
            Center(
              child: ElevatedButton.icon(
                icon: const Icon(Icons.refresh),
                label: const Text('Run analysis'),
                onPressed: () async {
                  final sm = ScaffoldMessenger.of(context);
                  sm.showSnackBar(
                    const SnackBar(
                      content: Text(
                        'We are working on your personalised AI-driven Risk Analysis.\n'
                        'We will let you know once the analysis is ready.',
                      ),
                      duration: Duration(seconds: 3),
                    ),
                  );

                  await context.read<ScenarioState>().fetchRiskResult(
                    userId: ov.userId,
                    start: ov.start,
                    horizon: ov.horizon,
                  );

                  if (!context.mounted) return;
                  sm.showSnackBar(
                    const SnackBar(
                      content: Text('✅ Risk analysis has been finalised!'),
                      duration: Duration(seconds: 3),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 12),

            // Fan chart
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Balance forecast',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    FanChart(
                      histDates: st.result!.histDates,
                      histBalances: st.result!.histBalances,
                      dates: st.result!.dates,
                      median: st.result!.median,
                      p5: st.result!.p5,
                      p95: st.result!.p95,
                      best: st.result!.best,
                      worst: st.result!.worst,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Insights & recommendations
            InsightBlock(paragraphs: st.result!.insights),
            const SizedBox(height: 12),
            RecommendationsBlock(bullets: st.result!.recommendations),
            const SizedBox(height: 16),

            // Download ZIP button
            Center(
              child: ElevatedButton.icon(
                icon: const Icon(Icons.download),
                label: const Text('Download Risk Analysis'),
                onPressed: () => api.downloadRiskAnalysis(
                  userId: ov.userId,
                  start: ov.start,
                  horizon: ov.horizon,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
