// lib/screens/chatbot_chat_screen.dart

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../models/chatbot.dart';
import '../widgets/chat_message.dart';

class ChatbotChatScreen extends StatefulWidget {
  final Chatbot chatbot;

  ChatbotChatScreen({required this.chatbot});

  @override
  _ChatbotChatScreenState createState() => _ChatbotChatScreenState();
}

class _ChatbotChatScreenState extends State<ChatbotChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isSending = false;

  // Replace with your local API endpoint
  final String apiUrl = 'http://localhost:8000/api/chat';

  Future<void> _sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isSending = true;
    });

    _controller.clear();

    try {
      // Send the message to the backend API for processing
      var response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'chatbot_id': widget.chatbot.id,
          'message': text,
        }),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        String reply = data['reply'];

        setState(() {
          _messages.add(ChatMessage(text: reply, isUser: false));
        });
      } else {
        setState(() {
          _messages.add(ChatMessage(text: 'Error: ${response.body}', isUser: false));
        });
      }
    } catch (e) {
      setState(() {
        _messages.add(ChatMessage(text: 'Error: ${e.toString()}', isUser: false));
      });
    } finally {
      setState(() {
        _isSending = false;
      });
    }
  }

  Widget _buildMessageList() {
    return ListView.builder(
      reverse: true,
      itemCount: _messages.length,
      itemBuilder: (context, index) {
        ChatMessage message = _messages[_messages.length - 1 - index];
        return ChatMessageWidget(message: message);
      },
    );
  }

  Widget _buildInputArea() {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8.0),
      color: Colors.white,
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: InputDecoration(
                hintText: 'Type your message...',
                border: InputBorder.none,
              ),
              onSubmitted: _isSending ? null : _sendMessage,
            ),
          ),
          IconButton(
            icon: _isSending ? CircularProgressIndicator() : Icon(Icons.send),
            onPressed: _isSending ? null : () => _sendMessage(_controller.text),
            color: Colors.blue,
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    String chatbotName = widget.chatbot.name;

    return Scaffold(
      appBar: AppBar(
        title: Text(chatbotName),
      ),
      body: Column(
        children: [
          Expanded(child: _buildMessageList()),
          Divider(height: 1.0),
          _buildInputArea(),
        ],
      ),
    );
  }
}
