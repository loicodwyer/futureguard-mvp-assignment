import 'package:flutter/material.dart';

/// Shows the “Insights” section - every sentence on its own line
/// and everything centre-aligned.
///
class InsightBlock extends StatelessWidget {
  final List<String> paragraphs;
  const InsightBlock({super.key, required this.paragraphs});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center, // ← centred
          children: [
            const Text(
              'Insights',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            // each sentence on its own row
            for (final p in paragraphs) ...[
              _styledText(p),
              const SizedBox(height: 8),
            ],
          ],
        ),
      ),
    );
  }

  /// Converts blocks to bold, keeps everything centre-aligned.
  Widget _styledText(String input) {
    final spans = <TextSpan>[];
    final exp = RegExp(r'\*(.*?)\*');
    int start = 0;

    for (final m in exp.allMatches(input)) {
      if (m.start > start) {
        spans.add(TextSpan(text: input.substring(start, m.start)));
      }
      spans.add(
        TextSpan(
          text: m.group(1),
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      );
      start = m.end;
    }
    if (start < input.length) {
      spans.add(TextSpan(text: input.substring(start)));
    }

    return Text.rich(TextSpan(children: spans), textAlign: TextAlign.center);
  }
}
