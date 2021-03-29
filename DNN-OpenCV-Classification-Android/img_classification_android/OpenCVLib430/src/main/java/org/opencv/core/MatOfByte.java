package org.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfByte extends Mat {
    // 8UC(x)
    private static final int _depth = CvType.CV_8U;
    private static final int _channels = 1;

    public MatOfByte() {
        super();
    }

    protected MatOfByte(long addr) {
        super(addr);
        if( !empty() && checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incompatible Mat");
        //FIXME: do we need release() here?
    }

    public static MatOfByte fromNativeAddr(long addr) {
        return new MatOfByte(addr);
    }

    public MatOfByte(Mat m) {
        super(m, Range.all());
        if( !empty() && checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incompatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfByte(byte...a) {
        super();
        fromArray(a);
    }

    public MatOfByte(int offset, int length, byte...a) {
        super();
        fromArray(offset, length, a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(byte...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length / _channels;
        alloc(num);
        put(0, 0, a); //TODO: check ret val!
    }

    public void fromArray(int offset, int length, byte...a) {
        if (offset < 0)
            throw new IllegalArgumentException("offset < 0");
        if (a == null)
            throw new NullPointerException();
        if (length < 0 || length + offset > a.length)
            throw new IllegalArgumentException("invalid 'length' parameter: " + Integer.toString(length));
        if (a.length == 0)
            return;
        int num = length / _channels;
        alloc(num);
        put(0, 0, a, offset, length); //TODO: check ret val!
    }

    public byte[] toArray() {
        int num = checkVector(_channels, _depth);
        if(num < 0)
            throw new RuntimeException("Native Mat has unexpected type or size: " + toString());
        byte[] a = new byte[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Byte> lb) {
        if(lb==null || lb.size()==0)
            return;
        Byte ab[] = lb.toArray(new Byte[0]);
        byte a[] = new byte[ab.length];
        for(int i=0; i<ab.length; i++)
            a[i] = ab[i];
        fromArray(a);
    }

    public List<Byte> toList() {
        byte[] a = toArray();
        Byte ab[] = new Byte[a.length];
        for(int i=0; i<a.length; i++)
            ab[i] = a[i];
        return Arrays.asList(ab);
    }
}
