import React, { useState, useCallback } from 'react';
import * as SecureStore from 'expo-secure-store';
import { FontAwesome, FontAwesome5, Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import {
    Image,
    ScrollView,
    StatusBar,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';


// Local images
import clinic1Image from '../assets/images/Enhance Vision Glasses.jpg';
import clinic2Image from '../assets/images/Enhance Vision Available  Glasses.jpg';
import clinic3Image from '../assets/images/Vision Training Room.jpg';
import clinic4Image from '../assets/images/Enhance Vision Machine.jpg';
import clinic5Image from '../assets/images/Enhance Vision Optical Clinic.jpg';
import clinic6Image from '../assets/images/Enhance Vision Clinic.jpg';
import dr1Image from '../assets/images/dr1.jpg';
import dr2Image from '../assets/images/dr2.jpg';

export default function DashboardScreen() {
    const router = useRouter();

    const [userData, setUserData] = useState({
        first_name: '',
        last_name: '',
        email: '',
        profile_image: null,
    });

    // Example: unread notifications count
    const [unreadCount, setUnreadCount] = useState(1); // Replace with actual fetch from API

    // Load user profile from SecureStore
    const loadLocalUserProfile = useCallback(async () => {
        try {
            const localProfileStr = await SecureStore.getItemAsync('userProfile');
            if (localProfileStr) {
                const localProfile = JSON.parse(localProfileStr);

                setUserData({
                    first_name: localProfile.first_name || 'Guest',
                    last_name: localProfile.last_name || '',
                    email: localProfile.email || '',
                    profile_image: localProfile.profile_image ?? null

                });
            } else {
                const token = await SecureStore.getItemAsync('userToken');
                if (!token) {
                    router.replace('/login');
                } else {
                    setUserData({
                        first_name: 'Guest',
                        last_name: '',
                        email: '',
                        profile_image: null,
                    });
                }
            }
        } catch (error) {
            console.warn('Failed to load local user profile:', error);
            const token = await SecureStore.getItemAsync('userToken');
            if (!token) router.replace('/login');
        }
    }, [router]);

    useFocusEffect(
        useCallback(() => {
            loadLocalUserProfile();
            return () => {};
        }, [loadLocalUserProfile])
    );

    const handleBookAppointment = () => router.push('/book-appointment');
    const handleDoctorsPress = () => router.push('/doctors');
    const handleProfilePress = () => router.push('/profile');
    const handleAppointmentsPress = () => router.push('/my-appointments');
    const handleNotificationsPress = () => {
        router.push('/notification');
        setUnreadCount(0); // Clear badge when user opens notifications
    };
    const handleBackPress = () => router.replace('/');

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="dark-content" />

            <ScrollView contentContainerStyle={styles.scrollViewContent}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity style={styles.circleBackButton} onPress={handleBackPress}>
                        <Ionicons name="arrow-back" size={20} color="#000" />
                    </TouchableOpacity>

                    <View style={styles.userInfo}>
                        {/* PROFILE IMAGE */}
                        <Image
                            source={{
                                uri: userData.profile_image
                                    ? userData.profile_image
                                    : 'https://cdn-icons-png.flaticon.com/512/706/706830.png',
                            }}
                            style={styles.profilePic}
                        />
                        <View style={styles.userNameRow}>
                            <View>
                                <Text style={styles.greetingText}>Hi, Welcome to SmartSight</Text>
                                <Text style={styles.userName}>
                                    {userData.first_name} {userData.last_name}
                                </Text>
                            </View>

                            <View style={styles.headerIcons}>
                                <TouchableOpacity style={styles.iconButton} onPress={handleNotificationsPress}>
                                    <View style={{ position: 'relative' }}>
                                        <MaterialIcons
                                            name="notifications"
                                            size={24}
                                            color={unreadCount > 0 ? '#77CDE0' : '#555'}
                                        />
                                        {unreadCount > 0 && (
                                            <View style={styles.badge}>
                                                <Text style={styles.badgeText}>{unreadCount}</Text>
                                            </View>
                                        )}
                                    </View>
                                </TouchableOpacity>
                                <TouchableOpacity style={styles.iconButton}>
                                    <Ionicons name="" size={24} color="#555" />
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>

                    <View style={styles.locationInfo}>
                        <MaterialIcons name="location-on" size={16} color="#888" />
                        <Text style={styles.locationText}>Limketkai, Lapasan</Text>
                    </View>
                </View>

                {/* Services Section */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.sectionTitle}>Services</Text>
                </View>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.serviceList}>
                    <TouchableOpacity style={styles.serviceItem} onPress={() => router.push('/eye-screening')}>
                        <View style={[styles.serviceIconContainer, { backgroundColor: '#FF8888' }]}>
                            <FontAwesome name="eye" size={30} color="#fff" />
                        </View>
                        <Text style={styles.serviceText}>Preliminary Eye Screening</Text>
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.serviceItem} onPress={handleBookAppointment}>
                        <View style={[styles.serviceIconContainer, { backgroundColor: '#77CDE0' }]}>
                            <FontAwesome5 name="calendar-check" size={30} color="#fff" />
                        </View>
                        <Text style={styles.serviceText}>Book Appointment Only</Text>
                    </TouchableOpacity>
                </ScrollView>

                {/* Available Doctors Section */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.sectionTitle}>Available Doctors</Text>
                    <TouchableOpacity onPress={handleDoctorsPress}>
                        <Text style={styles.seeAllText}>See All</Text>
                    </TouchableOpacity>
                </View>
                <View style={styles.doctorList}>
                    <TouchableOpacity style={styles.doctorCard}>
                        <Image source={dr2Image} style={styles.doctorImage} />
                        <View style={styles.doctorInfo}>
                            <Text style={styles.doctorName}>Dr. Mikaela Sherry Lopez</Text>
                            <Text style={styles.doctorAvailability}>Mon-Fri (10AM - 9PM)</Text>
                        </View>
                        <Text style={styles.doctorSpecialty}>Optometrist</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.doctorCard}>
                        <Image source={dr1Image} style={styles.doctorImage} />
                        <View style={styles.doctorInfo}>
                            <Text style={styles.doctorName}>Dr. Maria Cherry Lopez</Text>
                            <Text style={styles.doctorAvailability}>Mon-Fri (10AM - 9PM)</Text>
                        </View>
                        <Text style={styles.doctorSpecialty}>Optometrist</Text>
                    </TouchableOpacity>
                </View>

                {/* Clinics Overview */}
                <View style={styles.clinicsOverviewContainer}>
                    <View style={styles.sectionHeader}>
                        <Text style={styles.clinicsTitle}>Clinics Overview</Text>
                        <TouchableOpacity onPress={() => router.push('/packages')}>
                            <Text style={styles.seeAllText}>Our Packages</Text>
                        </TouchableOpacity>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.clinicList}>
                        {[clinic1Image, clinic2Image, clinic3Image, clinic4Image, clinic5Image, clinic6Image].map(
                            (img, index) => (
                                <TouchableOpacity key={index} style={styles.aestheticClinicCard}>
                                    <Image source={img} style={styles.aestheticClinicImage} />
                                    <View style={styles.cardContent}>
                                        <Text style={styles.aestheticClinicName}>Clinic {index + 1}</Text>
                                        <Text style={styles.clinicLocation}>Cagayan de Oro City</Text>
                                    </View>
                                </TouchableOpacity>
                            )
                        )}
                    </ScrollView>
                </View>
            </ScrollView>

            {/* Bottom Navigation */}
            <View style={styles.bottomNavigation}>
                <TouchableOpacity style={styles.navItem}>
                    <MaterialIcons name="home" size={24} color="#77CDE0" />
                    <Text style={[styles.navText, { color: '#77CDE0' }]}>Home</Text>
                </TouchableOpacity>

                <TouchableOpacity style={styles.navItem} onPress={handleAppointmentsPress}>
                    <MaterialIcons name="event-note" size={24} color="#888" />
                    <Text style={styles.navText}>Appointments</Text>
                </TouchableOpacity>

                <TouchableOpacity style={styles.navItem} onPress={handleNotificationsPress}>
                    <View style={{ position: 'relative' }}>
                        <MaterialIcons
                            name="notifications"
                            size={24}
                            color={unreadCount > 0 ? '#77CDE0' : '#888'}
                        />
                        {unreadCount > 0 && (
                            <View style={styles.badge}>
                                <Text style={styles.badgeText}>{unreadCount}</Text>
                            </View>
                        )}
                    </View>
                    <Text style={[styles.navText, { color: unreadCount > 0 ? '#77CDE0' : '#888' }]}>
                        Notifications
                    </Text>
                </TouchableOpacity>

                <TouchableOpacity style={styles.navItem} onPress={handleProfilePress}>
                    <MaterialIcons name="person" size={24} color="#888" />
                    <Text style={styles.navText}>Profile</Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#E8F7FF' }, // Changed background
    scrollViewContent: { padding: 20, paddingBottom: 100 },

    circleBackButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 10,
    },

    header: { marginBottom: 20 },
    userInfo: { flexDirection: 'row', alignItems: 'center', marginVertical: 15 },
    profilePic: {
        width: 80,
        height: 80,
        borderRadius: 40,
        marginRight: 12,
        borderWidth: 3,
        borderColor: '#65A3D5', // same as previous
    },

    greetingText: { fontSize: 14, color: '#555' },
    userName: { fontSize: 20, fontWeight: 'bold', color: '#222' },

    userNameRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        flex: 1
    },
    headerIcons: { flexDirection: 'row', alignItems: 'center' },
    iconButton: { marginLeft: 15 },

    locationInfo: { flexDirection: 'row', alignItems: 'center' },
    locationText: { fontSize: 14, color: '#555', marginLeft: 5 },

    sectionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 15,
    },
    sectionTitle: { fontSize: 20, fontWeight: 'bold', color: '#333' },
    seeAllText: { fontSize: 14, color: '#1E3A8A', fontWeight: 'bold' },

    serviceList: { marginBottom: 20 },
    serviceItem: { alignItems: 'center', width: 120, marginRight: 15 },
    serviceIconContainer: {
        width: 60,
        height: 60,
        borderRadius: 15,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 8,
    },
    serviceText: {
        fontSize: 12,
        fontWeight: '500',
        color: '#555',
        textAlign: 'center',
    },

    doctorList: { marginBottom: 20 },
    doctorCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#77CDE0', // updated background color
        borderRadius: 15,
        padding: 15,
        marginBottom: 10,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
    },
    doctorImage: {
        width: 60,
        height: 60,
        borderRadius: 30,
        marginRight: 15,
        borderWidth: 2,
        borderColor: '#fff',
    },
    doctorInfo: { flex: 1 },
    doctorName: { fontSize: 18, fontWeight: 'bold', color: '#fff' },
    doctorAvailability: { fontSize: 12, color: '#fff' },
    doctorSpecialty: { fontSize: 14, color: '#fff' },

    clinicsOverviewContainer: { marginBottom: 20 },
    clinicsTitle: { fontSize: 20, fontWeight: 'bold', color: '#333' },
    clinicList: { marginTop: 10 },
    aestheticClinicCard: {
        width: 250,
        borderRadius: 20,
        overflow: 'hidden',
        marginRight: 15,
        backgroundColor: '#FFD54F',
        elevation: 5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
    },
    aestheticClinicImage: {
        width: '100%',
        height: 150,
        resizeMode: 'cover',
    },
    cardContent: { padding: 15 },
    aestheticClinicName: { fontSize: 16, fontWeight: 'bold', color: '#333' },
    clinicLocation: { fontSize: 12, color: '#888', marginTop: 5 },

    bottomNavigation: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        backgroundColor: '#C8EAF7',
        borderTopWidth: 1,
        borderTopColor: '#FFFF0',
        paddingVertical: 10,
        paddingHorizontal: 5,
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        elevation: 8,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    navItem: { alignItems: 'center', padding: 5 },
    navText: { fontSize: 10, color: '#fff', marginTop: 3 },

    // Badge
    badge: {
        position: 'absolute',
        top: -5,
        right: -10,
        minWidth: 16,
        height: 16,
        borderRadius: 8,
        backgroundColor: '#77CDE0',
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 3,
    },
    badgeText: {
        color: '#fff',
        fontSize: 10,
        fontWeight: 'bold',
    },
});
