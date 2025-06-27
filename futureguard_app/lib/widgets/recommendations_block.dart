import 'package:flutter/material.dart';

/// Shows up to three bullet-point recommendations.
class RecommendationsBlock extends StatelessWidget {
  final List<String> bullets;
  const RecommendationsBlock({super.key, required this.bullets});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          // centred header; bullets still start-aligned
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const Text(
              'Recommendations',
              textAlign: TextAlign.center, // <- centred in the card
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            // bullet list
            for (final b in bullets) ...[
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('• '),
                  Expanded(child: Text(b)),
                ],
              ),
              const SizedBox(height: 6),
            ],
          ],
        ),
      ),
    );
  }
}
