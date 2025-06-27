import 'package:flutter/material.dart';
import 'overview_page.dart';
import 'risk_dashboard.dart';

class RootNav extends StatefulWidget {
  const RootNav({super.key});

  @override
  State<RootNav> createState() => _RootNavState();
}

class _RootNavState extends State<RootNav> {
  int _idx = 0;

  final _screens = const [
    OverviewPage(), // Account Overview Tab
    RiskDashboard(), // Risk Analysis Tab
    Placeholder(), // You can replace this later with a Settings page
  ];

  final _items = const [
    BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Overview'),
    BottomNavigationBarItem(icon: Icon(Icons.trending_up), label: 'Risk'),
    BottomNavigationBarItem(icon: Icon(Icons.more_horiz), label: 'More'),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(child: _screens[_idx]),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _idx,
        items: _items,
        onTap: (i) => setState(() => _idx = i),
      ),
    );
  }
}
