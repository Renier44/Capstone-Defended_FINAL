import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import {
    Image,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
    Alert,
} from 'react-native';
import { useState } from 'react';
import * as SecureStore from 'expo-secure-store';


export default function ConfirmBooking() {
    const router = useRouter();
    const params = useLocalSearchParams();

    const firstName = params.firstName || '';
    const lastName = params.lastName || '';
    const dob = params.dob || '';
    const gender = params.gender || '';
    const age = params.age || '';
    const reason = params.reason || '';
    const bookingFor = params.bookingFor || '';
    const selectedTime = params.selectedTime || '';
    const appointmentDate = params.appointmentDate || '';
    const doctorId = params.doctorId || '';
    const assignedDoctorName = params.assignedDoctorName || '';

    const [submitting, setSubmitting] = useState(false);

    // ✅ Format datetime exactly as backend expects: YYYY-MM-DD hh:mm AM/PM
    const formatDateTime = (date, time) => {
        return `${date} ${time}`; // ex: "2025-09-27 10:00 AM"
    };

const handleConfirm = async () => {
    setSubmitting(true);
    try {
        const full_datetime_str = formatDateTime(appointmentDate, selectedTime);
        

        const token = await SecureStore.getItemAsync('userToken');

const res = await fetch(
  'https://2b7bf55b1e09.ngrok-free.app/api/appointments/create/',
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Token ${token}` } : {})
    },
    body: JSON.stringify({
      firstName,
      lastName,
      age: parseInt(age),
      gender,
      reason,
      bookingFor,
      appointment_datetime_str: full_datetime_str,
      doctorId: parseInt(doctorId, 10),
    }),
  }
);


        const data = await res.json();

        if (!res.ok) {
            // If error (like timeslot unavailable), redirect with error message
            router.push({
                pathname: '/confirmation',
                params: {
                    success: 'false',
                    message: data.error || 'The selected timeslot is no longer available. Please choose another one.'
                }
            });
            return;
        }

        // ✅ Success: redirect with success message
        router.push({
            pathname: '/confirmation',
            params: {
                success: 'true',
                message: 'Your appointment has been submitted successfully.'
            }
        });

    } catch (err) {
        router.push({
            pathname: '/confirmation',
            params: {
                success: 'false',
                message: 'Network error. Could not reach the server.'
            }
        });
    } finally {
        setSubmitting(false);
    }
};


    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <Ionicons name="arrow-back" size={24} color="#2260FF" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Your Appointment</Text>
            </View>

            <View style={styles.doctorCard}>
                <Image
                    source={{
                        uri: `https://ui-avatars.com/api/?name=${assignedDoctorName}&background=2260FF&color=fff&size=60&bold=true`,
                    }}
                    style={styles.doctorImage}
                />
                <View>
                    <Text style={styles.doctorName}>{assignedDoctorName}</Text>
                    <Text style={styles.doctorTitle}>Optometrist</Text>
                </View>
            </View>

            <View style={styles.detailsContainer}>
                <View style={styles.detailRow}>
                    <Text style={styles.label}>Appointment Date:</Text>
                    <Text style={styles.detailText}>{appointmentDate || 'Not set'}</Text>
                </View>

                <View style={styles.detailRow}>
                    <Text style={styles.label}>Time:</Text>
                    <Text style={styles.detailText}>{selectedTime || 'Not set'}</Text>
                </View>

                <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Booking For</Text>
                    <Text style={styles.detailText}>
                        {bookingFor === 'yourself' ? 'Yourself' : bookingFor || 'N/A'}
                    </Text>

                    <Text style={styles.sectionTitle}>Last Name</Text>
                    <Text style={styles.detailText}>{lastName || 'N/A'}</Text>

                    <Text style={styles.sectionTitle}>First Name</Text>
                    <Text style={styles.detailText}>{firstName || 'N/A'}</Text>

                    <Text style={styles.sectionTitle}>Age</Text>
                    <Text style={styles.detailText}>{age || 'N/A'}</Text>

                    <Text style={styles.sectionTitle}>Gender</Text>
                    <Text style={styles.detailText}>{gender || 'N/A'}</Text>

                    <Text style={styles.sectionTitle}>Reason of Appointment</Text>
                    <Text style={styles.detailText}>{reason || 'N/A'}</Text>
                </View>
            </View>

            <TouchableOpacity
                style={[styles.confirmButton, submitting && { opacity: 0.6 }]}
                onPress={handleConfirm}
                disabled={submitting}
            >
                <Text style={styles.confirmButtonText}>
                    {submitting ? 'Submitting…' : 'Confirm Booking'}
                </Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    // Note: Custom fonts in React Native require additional steps (like adding the font file to the project)
    // which is not possible in this single-file environment. The 'LeagueSpartan' font family is added
    // here to demonstrate where the change would be applied if the font were available.
    container: { flex: 1, backgroundColor: '#C8EAF7' },
    content: { padding: 20 },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        paddingTop: 40,
        backgroundColor: '#C8EAF7',
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        marginLeft: 16,
        color: '#2260FF',
        fontFamily: 'LeagueSpartan',
        flex: 1,
        textAlign: 'center',
    },
    doctorCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFD54F',
        padding: 15,
        borderRadius: 15,
        marginBottom: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    doctorImage: { width: 60, height: 60, borderRadius: 30, marginRight: 15 },
    doctorName: { fontSize: 16, fontWeight: 'bold', color: '#2260FF', fontFamily: 'LeagueSpartan' },
    doctorTitle: { fontSize: 14, color: '#555', fontFamily: 'LeagueSpartan' },
    detailsContainer: {
        backgroundColor: '#D6ECFF',
        borderRadius: 15,
        padding: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    detailRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 10,
    },
    label: { fontWeight: 'bold', fontFamily: 'LeagueSpartan' },
    section: { marginTop: 20 },
    sectionTitle: { fontWeight: 'bold', marginTop: 10, color: '#2260FF', fontFamily: 'LeagueSpartan' },
    detailText: { marginBottom: 10, fontSize: 14, color: '#333', fontFamily: 'LeagueSpartan' },
    confirmButton: {
        backgroundColor: '#2260FF',
        padding: 15,
        borderRadius: 10,
        marginTop: 30,
        marginBottom: 20,
        shadowColor: '#000',
        shadowOpacity: 0.15,
        shadowOffset: { width: 0, height: 2 },
        shadowRadius: 4,
        elevation: 3,
    },
    confirmButtonText: { textAlign: 'center', fontWeight: 'bold', color: '#fff', fontFamily: 'LeagueSpartan' },
});
