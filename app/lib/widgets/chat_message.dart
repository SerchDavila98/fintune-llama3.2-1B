// lib/widgets/chat_message.dart

import 'package:flutter/material.dart';

class ChatMessage {
  final String text;
  final bool isUser;

  ChatMessage({required this.text, required this.isUser});
}

class ChatMessageWidget extends StatelessWidget {
  final ChatMessage message;

  ChatMessageWidget({required this.message});

  @override
  Widget build(BuildContext context) {
    Alignment alignment = message.isUser ? Alignment.centerRight : Alignment.centerLeft;
    Color color = message.isUser ? Colors.blue[100]! : Colors.grey[300]!;

    return Container(
      alignment: alignment,
      padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 4.0),
      child: Container(
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(12.0),
        ),
        padding: EdgeInsets.all(12.0),
        child: Text(message.text),
      ),
    );
  }
}
