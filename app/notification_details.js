import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ActivityIndicator,
  ScrollView,
  TouchableOpacity,
  StatusBar
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { useRouter, useLocalSearchParams } from 'expo-router';

const API_BASE = 'https://capstone-defended-final.onrender.com';
const NOTIFICATION_DETAIL_ENDPOINT = '/api/notification/'; // Append ID
const MARK_READ_ENDPOINT = '/api/notification/mark-read/'; // POST ID

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

export default function NotificationDetails() {
  const router = useRouter();
  const { id } = useLocalSearchParams(); // Get notification ID from params
  const [notification, setNotification] = useState(null);
  const [loading, setLoading] = useState(true);
  const [markingRead, setMarkingRead] = useState(false);

  const fetchNotificationDetail = async () => {
    setLoading(true);
    try {
      const token = await SecureStore.getItemAsync('userToken');
      if (!token) return;

      const res = await fetch(`${API_BASE}${NOTIFICATION_DETAIL_ENDPOINT}${id}/`, {
        headers: { Authorization: `Token ${token}` },
      });

      if (!res.ok) throw new Error('Failed to fetch notification details');

      const data = await res.json();
      setNotification(data);
    } catch (err) {
      console.error('Fetch notification error:', err);
    } finally {
      setLoading(false);
    }
  };

  const markAsRead = async () => {
    if (!notification || notification.is_read) return;
    setMarkingRead(true);
    try {
      const token = await SecureStore.getItemAsync('userToken');
      if (!token) return;

      const res = await fetch(`${API_BASE}${MARK_READ_ENDPOINT}${id}/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Token ${token}` },
      });

      if (res.ok) {
        setNotification(prev => ({ ...prev, is_read: true }));
      } else {
        console.error('Mark as read failed');
      }
    } catch (err) {
      console.error(err);
    } finally {
      setMarkingRead(false);
    }
  };

  useEffect(() => {
    fetchNotificationDetail();
  }, [id]);

  if (loading) {
    return (
      <SafeAreaView style={styles.centered}>
        <ActivityIndicator size="large" color="#1877F2" />
        <Text style={{ marginTop: 8, color: '#555' }}>Loading notification...</Text>
      </SafeAreaView>
    );
  }

  if (!notification) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={{ color: '#555' }}>Notification not found.</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#E8F5FB" />
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#1877F2" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notification Details</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <Text style={styles.title}>{notification.title}</Text>
          <Text style={styles.message}>{notification.message}</Text>
          {notification.extra_info && (
            <Text style={styles.extra}>{notification.extra_info}</Text>
          )}
          <Text style={styles.date}>
            Sent at: {new Date(notification.created_at).toLocaleString()}
          </Text>
        </View>

        {/* Mark as Read Button */}
        {!notification.is_read && (
          <TouchableOpacity
            style={styles.markButton}
            onPress={markAsRead}
            disabled={markingRead}
          >
            <Text style={styles.markButtonText}>
              {markingRead ? 'Marking...' : 'Mark as Read'}
            </Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  // --- Layout & Loading ---
  container: { 
    flex: 1, 
    backgroundColor: BACKGROUND_COLOR 
},
  centered: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center', 
    backgroundColor: BACKGROUND_COLOR 
},
  loadingText: { 
    marginTop: 8, 
    color: BRAND_BLUE,
    fontFamily: "VarelaRound-Regular" // FIX: Ensure quotes around font names
  },
  errorText: { 
    color: NEUTRAL_TEXT,
    fontFamily: "VarelaRound-Regular" // FIX: Ensure quotes around font names
  },


  // --- Header ---
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: BACKGROUND_COLOR,
    borderBottomWidth: 1,
    borderBottomColor: INFO_BORDER_COLOR,
    ...CARD_ELEVATION,
  },
  backButton: { 
    padding: 5, 
    marginRight: 10 
},
  headerTitle: { 
    fontSize: 22, 
    fontWeight: '800', 
    color: BRAND_BLUE, 
    flex: 1, 
    textAlign: 'center',
    fontFamily: "Montserrat-VariableFont_wght", // FIX: Ensure quotes around font names
},
  headerSpacer: {
    width: 34, // Matches backButton width+margin for centering title
  },

  // --- Content & Card ---
  content: { 
    padding: 20, 
    alignItems: 'center' 
},
  card: {
    width: '100%',
    backgroundColor: CARD_BG,
    borderRadius: 15,
    padding: 25,
    borderLeftWidth: 5,
    borderLeftColor: PRIMARY_ACTION_COLOR, // Yellow accent border
    ...CARD_ELEVATION,
    marginBottom: 30,
  },
  title: { 
    fontSize: 24, 
    fontWeight: '900', 
    marginBottom: 15, 
    color: BRAND_BLUE, 
    textAlign: 'center',
    fontFamily: "Montserrat-VariableFont_wght", // FIX: Ensure quotes around font names
},
  message: { 
    fontSize: 16, 
    color: NEUTRAL_TEXT, 
    marginBottom: 15, 
    textAlign: 'left', 
    lineHeight: 24,
    fontFamily: "VarelaRound-Regular", // FIX: Ensure quotes around font names
},
  extra: { 
    fontSize: 14, 
    color: '#777', 
    marginBottom: 15, 
    textAlign: 'left',
    fontFamily: "VarelaRound-Regular", // FIX: Ensure quotes around font names
},
  date: { 
    fontSize: 12, 
    color: '#999', 
    marginTop: 10, 
    textAlign: 'right',
    fontFamily: "VarelaRound-Regular", // FIX: Ensure quotes around font names
},

  // --- Action Button ---
  markButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: PRIMARY_ACTION_COLOR, 
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    width: '70%',
    ...CARD_ELEVATION,
  },
  markButtonText: { 
    color: BRAND_BLUE, 
    fontSize: 18, 
    fontWeight: '800',
    fontFamily: "Montserrat-VariableFont_wght", // FIX: Ensure quotes around font names
},
});