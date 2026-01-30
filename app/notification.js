import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, StyleSheet, FlatList,  ActivityIndicator,
  TouchableOpacity, RefreshControl, StatusBar
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { useRouter, useFocusEffect } from 'expo-router';


const API_BASE = 'https://capstone-defended-final.onrender.com';
const NOTIFICATIONS_ENDPOINT = '/api/notification/';
const MARK_ALL_READ_ENDPOINT = '/api/notification/mark-all-read/';

// 1. GLOBAL CONSTANTS (Matching your app's theme)
// =======================================================
const BRAND_BLUE = "#0057B7"; 
const PRIMARY_ACTION_COLOR = "#FFD54F"; // Yellow accent
const BACKGROUND_COLOR = "#E8F7FF"; // Light blue background
const NEUTRAL_TEXT = "#333333"; 
const INFO_BORDER_COLOR = "#77CDE0"; // Light blue border/accent color
const CARD_BG = "#fff"; // Clean white background for cards

const CARD_ELEVATION = {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 3,
};

export default function NotificationScreen() {
  const router = useRouter();
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const fetchNotifications = useCallback(async () => {
    setLoading(true);
    try {
      const token = await SecureStore.getItemAsync('userToken');
      if (!token) return;

      const res = await fetch(`${API_BASE}${NOTIFICATIONS_ENDPOINT}`, {
        headers: { Authorization: `Token ${token}` },
      });
      const data = await res.json();
      setNotifications(data);
    } catch (e) {
      console.error('Fetch error:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const markAllAsRead = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      if (!token) return;

      const res = await fetch(`${API_BASE}${MARK_ALL_READ_ENDPOINT}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Token ${token}` },
      });

      if (res.ok) {
        setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
      }
    } catch (e) {
      console.error('Mark all read failed:', e);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    fetchNotifications();
  };

  useFocusEffect(useCallback(() => {
    fetchNotifications();
  }, [fetchNotifications]));

  const formatTime = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return 'Date unavailable';
    }
  };

  const handleNotificationPress = (item) => {
    // Optionally mark the notification as read immediately
    if (!item.is_read) {
      setNotifications(prev => prev.map(n =>
        n.id === item.id ? { ...n, is_read: true } : n
      ));
    }

    // Navigate to notification_details and pass the notification ID
    router.push({
      pathname: '/notification_details',
      params: { id: item.id.toString() },
    });
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={[styles.notificationCard, !item.is_read && styles.unreadCard]}
      activeOpacity={0.8}
      onPress={() => handleNotificationPress(item)}
    >
      <View style={styles.iconContainer}>
        {!item.is_read && <View style={styles.unreadDot} />}
      </View>
      <View style={styles.textContainer}>
        <Text style={[styles.titleText, !item.is_read && styles.unreadTitle]}>{item.title}</Text>
        <Text style={styles.messageText}>{item.message}</Text>
        <Text style={styles.timeText}>{formatTime(item.created_at)}</Text>
      </View>
    </TouchableOpacity>
  );

  const displayedNotifications = showAll ? notifications : notifications.slice(0, 7);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#E8F5FB" />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#1877F2" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notifications</Text>
        <TouchableOpacity onPress={markAllAsRead} style={styles.markAllButton}>
          <Text style={styles.markAllText}>Mark all</Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color="#1877F2" />
          <Text style={{ color: '#777', marginTop: 8 }}>Loading notifications...</Text>
        </View>
      ) : (
        <>
          <FlatList
            data={displayedNotifications}
            keyExtractor={(item) => item.id.toString()}
            renderItem={renderItem}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} tintColor="#65A3D5" />
            }
            ListEmptyComponent={() => (
              <View style={styles.emptyContainer}>
                <MaterialIcons name="notifications-none" size={60} color="#B0BEC5" />
                <Text style={styles.emptyText}> You&apos;re all caught up! </Text>
              </View>
            )}
            contentContainerStyle={styles.listContainer}
          />

          {notifications.length > 7 && !showAll && (
            <TouchableOpacity style={styles.seeMoreButton} onPress={() => setShowAll(true)}>
              <Text style={styles.seeMoreText}>See previous notifications</Text>
            </TouchableOpacity>
          )}
        </>
      )}
    </SafeAreaView>
  );
}

// 🌈 STYLES (UPDATED)
const styles = StyleSheet.create({
    // --- Overall Layout ---
    container: { 
        flex: 1, 
        backgroundColor: BACKGROUND_COLOR, // Light blue background
    },
    listContainer: { 
        paddingHorizontal: 20, 
        paddingTop: 10,
        paddingBottom: 25 
    },
    centered: { 
        flex: 1, 
        justifyContent: 'center', 
        alignItems: 'center' 
    },
    loadingText: {
        color: BRAND_BLUE, 
        marginTop: 8,
        fontFamily: "VarelaRound-Regular",
    },

    // --- Header ---
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 12,
        backgroundColor: BACKGROUND_COLOR, // White header background
        borderBottomWidth: 1,
        borderBottomColor: INFO_BORDER_COLOR,
        ...CARD_ELEVATION, // Subtle shadow for header
    },
    headerTitle: { 
        fontSize: 22, 
        fontWeight: '800', 
        color: BRAND_BLUE, // Main blue color
        flex: 1, 
        textAlign: 'center',
        fontFamily: "Montserrat-VariableFont_wght",
    },
    backButton: { 
        padding: 5, // Keep spacing for touch target
    },
    markAllButton: { 
        paddingHorizontal: 5,
        paddingVertical: 5,
    },
    markAllText: { 
        fontSize: 14, 
        color: BRAND_BLUE, 
        fontWeight: '700',
        fontFamily: "VarelaRound-Regular",
    },

    // --- Notification Card ---
    notificationCard: {
        flexDirection: 'row',
        backgroundColor: CARD_BG,
        borderRadius: 15, // Rounded corners for cards
        padding: 15,
        marginBottom: 10,
        ...CARD_ELEVATION, // Consistent card elevation
        borderLeftWidth: 5,
        borderLeftColor: INFO_BORDER_COLOR, // Accent color border
    },
    unreadCard: {
        backgroundColor: '#F3F9FF', // Slightly different background for unread
        borderLeftColor: PRIMARY_ACTION_COLOR, // Yellow accent for unread
    },
    iconContainer: { 
        width: 25, 
        alignItems: 'center', 
        paddingTop: 4, 
        marginRight: 10 
    },
    unreadDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: BRAND_BLUE, // Blue dot for unread
    },
    textContainer: { 
        flex: 1 
    },
    titleText: { 
        fontSize: 16, 
        color: NEUTRAL_TEXT, 
        fontWeight: '600',
        fontFamily: "Montserrat-VariableFont_wght",
    },
    unreadTitle: { 
        color: BRAND_BLUE, 
        fontWeight: '900',
    },
    messageText: { 
        fontSize: 14, 
        color: '#555', 
        marginTop: 4,
        fontFamily: "VarelaRound-Regular",
    },
    timeText: { 
        fontSize: 12, 
        color: '#888', 
        marginTop: 6,
        fontFamily: "VarelaRound-Regular",
    },

    // --- See More Button ---
    seeMoreButton: {
        paddingVertical: 14,
        marginHorizontal: 20,
        borderRadius: 12,
        backgroundColor: PRIMARY_ACTION_COLOR, // Yellow action button
        alignItems: 'center',
        marginTop: 5,
        marginBottom: 20,
        ...CARD_ELEVATION,
    },
    seeMoreText: {
        color: BRAND_BLUE, // Blue text on yellow button
        fontWeight: '800',
        fontSize: 16,
        fontFamily: "Montserrat-VariableFont_wght",
    },
    
    // --- Empty State ---
    emptyContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: 100,
    },
    emptyText: {
        fontSize: 16,
        color: NEUTRAL_TEXT,
        marginTop: 10,
        fontWeight: '500',
        fontFamily: "VarelaRound-Regular",
    },
});