// lib/widgets/app_frame.dart
//
// Constrains the whole app to a typical phone size when
// you run the web build in a desktop browser.
// ---------------------------------------------------------------------

import 'package:flutter/material.dart';

class AppFrame extends StatelessWidget {
  const AppFrame({super.key, required this.child});

  // child = the widget tree produced by MaterialApp
  final Widget child;

  @override
  Widget build(BuildContext context) {
    // 430×932 is the logical size of many modern phones.
    const double maxW = 430;
    const double maxH = 932;

    return Container(
      color: const Color(0xFF121212), // dark gray background
      alignment: Alignment.topCenter, // center the phone
      child: FittedBox(
        // avoid pixel stretching
        alignment: Alignment.topCenter,
        child: SizedBox(
          width: maxW,
          height: maxH,
          child: ClipRRect(
            // rounded corners
            borderRadius: BorderRadius.circular(24),
            child: Material(
              // gives elevation & shadow
              elevation: 4,
              shadowColor: Colors.black26,
              child: child, // <-- your real app
            ),
          ),
        ),
      ),
    );
  }
}
