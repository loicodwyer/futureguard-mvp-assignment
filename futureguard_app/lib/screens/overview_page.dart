import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/overview_state.dart';
import '../widgets/tx_tile.dart';

class OverviewPage extends StatelessWidget {
  const OverviewPage({super.key});

  @override
  Widget build(BuildContext context) {
    final st = context.watch<OverviewState>();

    // top row with 3 controls
    Widget controls() {
      // Build list of userId dropdowns from available data or fallback
      final availableIds = [
        for (int i = 1; i <= 100; i++) i,
      ]; // or change to a smaller range if needed

      return Row(
        children: [
          // user-id
          DropdownButton<int>(
            value: st.userId,
            items: availableIds
                .map((e) => DropdownMenuItem(value: e, child: Text('ID $e')))
                .toList(),
            onChanged: (v) {
              if (v == null) return;
              context.read<OverviewState>()
                ..userId = v
                ..fetch();
            },
          ),
          const SizedBox(width: 12),

          // start-date
          TextButton.icon(
            icon: const Icon(Icons.date_range),
            label: Text(
              '${st.start.year}-${st.start.month.toString().padLeft(2, '0')}-${st.start.day.toString().padLeft(2, '0')}',
            ),
            onPressed: () async {
              final picked = await showDatePicker(
                context: context,
                initialDate: st.start,
                firstDate: DateTime(2024, 1, 1),
                lastDate: DateTime(2026, 12, 31),
              );
              if (picked != null) {
                context.read<OverviewState>()
                  ..start = picked
                  ..fetch();
              }
            },
          ),
          const SizedBox(width: 12),

          // horizon
          DropdownButton<int>(
            value: st.horizon,
            items: const [30, 61, 92, 183, 365]
                .map((e) => DropdownMenuItem(value: e, child: Text('$e d')))
                .toList(),
            onChanged: (v) {
              if (v == null) return;
              context.read<OverviewState>()
                ..horizon = v
                ..fetch();
            },
          ),
        ],
      );
    }

    // main body
    Widget body() {
      if (st.busy) return const Center(child: CircularProgressIndicator());
      if (st.error != null) return Center(child: Text('❌ ${st.error}'));
      if (st.data == null)
        return const Center(child: Text('Choose any filter…'));

      final d = st.data!;

      // 4 little buttons (“widgets”) – just show them as chips for now
      List<Widget> widgetChips = d.widgets
          .map((w) => Chip(label: Text(w)))
          .toList();

      return ListView(
        padding: const EdgeInsets.all(12),
        children: [
          // header card  (name; iban; balance)
          Card(
            color: Theme.of(context).colorScheme.surfaceVariant.withOpacity(.3),
            child: ListTile(
              title: Text(
                d.name.isEmpty ? '—' : d.name,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              subtitle: Text(d.iban.isEmpty ? '—' : d.iban),
              trailing: Text(
                '€${d.balance.toStringAsFixed(2)}',
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),

          // 4 “action” widgets
          Wrap(spacing: 8, runSpacing: 4, children: widgetChips),
          const Divider(height: 24),

          // transactions list
          const Text(
            'Transactions',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 6),
          ...d.tx.map((t) => TxTile(tx: t)),
        ],
      );
    }

    return SafeArea(
      child: Column(
        children: [
          Padding(padding: const EdgeInsets.all(8), child: controls()),
          Expanded(child: body()),
        ],
      ),
    );
  }
}
