import 'package:flutter/material.dart';
import '../models/account_overview.dart';

class TxTile extends StatelessWidget {
  final Tx tx;
  const TxTile({super.key, required this.tx});

  Color _c() => tx.amount >= 0 ? Colors.green : Colors.red;

  @override
  Widget build(BuildContext context) => ListTile(
    dense: true,
    leading: Icon(Icons.swap_horiz, color: _c()),
    title: Text(tx.counterparty),
    subtitle: Text(
      '${tx.date.day.toString().padLeft(2, '0')}-'
      '${tx.date.month.toString().padLeft(2, '0')}-'
      '${tx.date.year}',
    ),
    trailing: Text(
      (tx.amount >= 0 ? '+€' : '€') + tx.amount.abs().toStringAsFixed(2),
      style: TextStyle(color: _c(), fontWeight: FontWeight.bold),
    ),
  );
}
