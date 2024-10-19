// lib/providers/chatbot_provider.dart

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/chatbot.dart';

class ChatbotProvider with ChangeNotifier {
  List<Chatbot> _chatbots = [];

  List<Chatbot> get chatbots => _chatbots;

  ChatbotProvider() {
    loadChatbots();
  }

  Future<void> loadChatbots() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? chatbotsString = prefs.getString('chatbots');
    if (chatbotsString != null) {
      _chatbots = Chatbot.decode(chatbotsString);
      notifyListeners();
    }
  }

  Future<void> addChatbot(Chatbot chatbot) async {
    _chatbots.add(chatbot);
    await saveChatbots();
    notifyListeners();
  }

  Future<void> removeChatbot(String id) async {
    _chatbots.removeWhere((chatbot) => chatbot.id == id);
    await saveChatbots();
    notifyListeners();
  }

  Future<void> updateChatbot(Chatbot updatedChatbot) async {
    int index = _chatbots.indexWhere((c) => c.id == updatedChatbot.id);
    if (index != -1) {
      _chatbots[index] = updatedChatbot;
      await saveChatbots();
      notifyListeners();
    }
  }

  Future<void> saveChatbots() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setString('chatbots', Chatbot.encode(_chatbots));
  }
}
