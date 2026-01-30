import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    FlatList,
    SafeAreaView,
    ActivityIndicator,
    TouchableOpacity,
    RefreshControl,
    StatusBar,
} from 'react-native';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import { useRouter, useFocusEffect } from 'expo-router';

// ✅ Base API URL (make sure your ngrok or Django URL is correct)
const API_BASE = 'https://2b7bf55b1e09.ngrok-free.app';
const NOTIFICATIONS_ENDPOINT = '/api/notifications/';

export default function NotificationScreen() {
    const router = useRouter(); // ✅ fixed navigation hook
    const [notifications, setNotifications] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);

    // 🔹 Fetch notifications from backend
    const fetchNotifications = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const userToken = await SecureStore.getItemAsync('userToken');
            if (!userToken) {
                setError('User not authenticated. Please log in.');
                setLoading(false);
                return;
            }

            const finalUrl = `${API_BASE}${NOTIFICATIONS_ENDPOINT}`;
            const response = await fetch(finalUrl, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Token ${userToken}`,
                },
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Server Error' }));
                throw new Error(errorData.detail || `Failed to fetch notifications: ${response.status}`);
            }

            const data = await response.json();
            setNotifications(data);
        } catch (e) {
            console.error('Error fetching notifications:', e.message);
            setError('Could not load notifications. Check your network or backend.');
            setNotifications([]);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    // 🔹 Pull-to-refresh handler
    const handleRefresh = () => {
        setRefreshing(true);
        fetchNotifications();
    };

    // 🔹 Fetch on screen focus
    useFocusEffect(
        useCallback(() => {
            fetchNotifications();
            return () => {};
        }, [fetchNotifications])
    );

    // 🔹 Format timestamp
    const formatTime = (timestamp) => {
        try {
            return new Date(timestamp).toLocaleString();
        } catch {
            return 'Date unavailable';
        }
    };

    // 🔹 Render a single notification item
    const renderItem = ({ item }) => (
        <TouchableOpacity
            style={[styles.notificationCard, item.is_read ? styles.read : styles.unread]}
            onPress={() => console.log('Tapped notification ID:', item.id)}
        >
            <View style={styles.iconContainer}>
                <MaterialIcons
                    name={item.is_read ? 'done-all' : 'circle-notifications'}
                    size={24}
                    color={item.is_read ? '#888' : '#65A3D5'}
                />
            </View>
            <View style={styles.textContainer}>
                <Text style={styles.titleText}>{item.title}</Text>
                <Text style={styles.messageText}>{item.message}</Text>
                <Text style={styles.timeText}>{formatTime(item.created_at)}</Text>
            </View>
        </TouchableOpacity>
    );

    // 🔹 Loading view
    if (loading && !refreshing) {
        return (
            <View style={styles.centered}>
                <ActivityIndicator size="large" color="#65A3D5" />
                <Text style={{ marginTop: 10 }}>Loading notifications...</Text>
            </View>
        );
    }

    // 🔹 Error view
    if (error) {
        return (
            <View style={styles.centered}>
                <Text style={styles.errorText}>{error}</Text>
                <TouchableOpacity style={styles.retryButton} onPress={handleRefresh}>
                    <Text style={styles.retryText}>Try Again</Text>
                </TouchableOpacity>
            </View>
        );
    }

    // 🔹 Main UI
    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="dark-content" />
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="#333" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Notification History</Text>
            </View>

            <FlatList
                data={notifications}
                keyExtractor={(item) => item.id.toString()}
                renderItem={renderItem}
                ListEmptyComponent={() => (
                    <View style={styles.emptyContainer}>
                        <MaterialIcons name="inbox" size={50} color="#ccc" />
                        <Text style={styles.emptyText}>No notifications yet.</Text>
                        <Text style={styles.emptySubText}>Book an appointment to see updates here.</Text>
                    </View>
                )}
                contentContainerStyle={notifications.length === 0 ? styles.listContainerEmpty : styles.listContainer}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} tintColor="#65A3D5" />
                }
            />
        </SafeAreaView>
    );
}

// ✅ Styles
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#fff' },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 15,
        backgroundColor: '#f8f8f8',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    backButton: { marginRight: 15 },
    headerTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#333',
    },
    listContainer: { paddingHorizontal: 15, paddingVertical: 10 },
    listContainerEmpty: { flexGrow: 1, justifyContent: 'center' },
    centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },

    notificationCard: {
        flexDirection: 'row',
        padding: 15,
        borderRadius: 10,
        marginBottom: 8,
        backgroundColor: '#fff',
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 1.5,
        alignItems: 'flex-start',
    },
    unread: { borderLeftWidth: 4, borderLeftColor: '#77CDE0' },
    read: { borderLeftWidth: 4, borderLeftColor: '#f0f0f0' },

    iconContainer: { marginRight: 15, paddingTop: 3 },
    textContainer: { flex: 1 },

    titleText: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 2,
    },
    messageText: {
        fontSize: 14,
        color: '#555',
        marginBottom: 5,
    },
    timeText: {
        fontSize: 12,
        color: '#888',
    },

    emptyContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40,
        height: 300,
    },
    emptyText: {
        fontSize: 18,
        color: '#888',
        marginTop: 10,
        fontWeight: '600',
    },
    emptySubText: {
        fontSize: 14,
        color: '#aaa',
        marginTop: 5,
    },
    errorText: {
        fontSize: 16,
        color: 'red',
        textAlign: 'center',
        marginBottom: 15,
    },
    retryButton: {
        backgroundColor: '#65A3D5',
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 8,
    },
    retryText: {
        color: '#fff',
        fontWeight: 'bold',
    },
});
