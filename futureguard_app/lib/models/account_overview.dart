// lib/models/account_overview.dart
//
//  Matches the JSON coming from  /account_overview/{user_id}
//  ---------------------------------------------------------------------

class AccountOverview {
  // header
  final int userId;
  final String name;
  final String iban;
  final double balance; // running balance up-to <start>

  // purple action buttons
  final List<String> widgets;

  // transactions shown in the list
  final List<Tx> tx;

  AccountOverview({
    required this.userId,
    required this.name,
    required this.iban,
    required this.balance,
    required this.widgets,
    required this.tx,
  });

  factory AccountOverview.fromJson(Map<String, dynamic> j) {
    // the API packs id / name / iban under the "user" key
    final u = j['user'] as Map<String, dynamic>? ?? {};

    return AccountOverview(
      userId: (u['id'] ?? 0) as int,
      name: (u['name'] ?? 'Unnamed').toString(),
      iban: (u['iban'] ?? '—').toString(),
      balance: (j['balance'] ?? 0).toDouble(),
      widgets: (j['widgets'] as List<dynamic>? ?? [])
          .map((e) => e.toString())
          .toList(),
      tx: (j['tx'] as List<dynamic>? ?? []).map((e) => Tx.fromJson(e)).toList(),
    );
  }
}

//  Tx helper
class Tx {
  final DateTime date;
  final String counterparty;
  final double amount; // +income / –expense

  Tx({required this.date, required this.counterparty, required this.amount});

  factory Tx.fromJson(Map<String, dynamic> j) => Tx(
    date: DateTime.tryParse(j['date']?.toString() ?? '') ?? DateTime(2000),
    counterparty: (j['counterparty'] ?? '(unknown)').toString(),
    amount: (j['amount'] ?? 0).toDouble(),
  );
}
